// CSPP (Crash-Safe Persistent Patricia) memtable — port of ToplingDB's
// cspp_memtable.cc (now cspp.cc) to vanilla FRocksDB-8.10.0's MemTableRep ABI.
//
// Adaptations from the upstream ToplingDB source:
//   - Implements the vanilla Insert(KeyHandle)/Get(LookupKey,cb)/Contains(const char*)
//     ABI instead of the fork's InsertKeyValue/KeyValuePair callback ABI.
//   - Drops the "memtable-as-log-index" feature (KeyValueToLogRef / KV_ToShortLogRef
//     and the m_wals array): vanilla 8.10.0 has no matching WAL-side patch.
//   - Drops in-place ConvertToSST and CSPPMemTabTable* (would need topling-sst).
//   - Drops ToplingDB-only MemTableRep extensions: OnDupUserKeyYield, ColdizeMemory,
//     ApproximateKeyAnchors, GetRandomInternalKeysAppend, InitSetMemTableAsLogIndex,
//     ToWebViewJson.
//   - Uses the standard ObjectLibrary registration (memtablerep.cc) instead of
//     rockside's ROCKSDB_REG_Plugin/JSON SidePluginRepo registration.
//
// What remains: the patricia trie itself (third-party/terark/fsa/cspptrie.*) plus
// the per-user-key Entry vector (VecPin) that stores tag→value pairs in the trie's
// own memory pool.

#include "rocksdb/memtablerep.h"

#include "db/memtable.h"
#include "memory/allocator.h"
#include "rocksdb/comparator.h"
#include "rocksdb/slice.h"
#include "rocksdb/utilities/options_type.h"
#include "rocksdb/write_buffer_manager.h"
#include "util/coding.h"

#if defined(_MSC_VER)
  #pragma warning(disable: 4245)
  #pragma warning(disable: 4458)
#endif
#include <terark/fsa/cspptrie.inl>
#include <terark/num_to_str.hpp>

#include <atomic>
#include <mutex>
#include <thread>

namespace ROCKSDB_NAMESPACE {

using terark::byte_t;
using terark::fstring;
using terark::MainPatricia;
using terark::Patricia;
using terark::as_atomic;
using terark::pow2_align_up;
using terark::lower_bound_0;
using terark::upper_bound_0;
using terark::binary_search_0;
// terark_bsr_u32 is a macro (preprocessor), so it can't be brought in with using.

namespace {

constexpr uint32_t LOCK_FLAG = uint32_t(1) << 31;

inline char* EncodeVarint32Ptr(char* dst, uint32_t v) {
  return EncodeVarint32(dst, v);
}

// Buffer encoding used by vanilla MemTableRep:
//   | varint32 internal_key_len | internal_key (user_key + 8-byte tag) |
//   | varint32 value_len        | value                                |
// `buf` must have capacity ≥ EncodedEntryLen(uk.size, val.size).
inline size_t EncodedEntryLen(size_t uk_size, size_t val_size) {
  const size_t klen = uk_size + 8;
  return VarintLength(klen) + klen + VarintLength(val_size) + val_size;
}

inline char* WriteEncodedEntry(char* dst, const Slice& user_key,
                               uint64_t tag, const Slice& value) {
  const size_t klen = user_key.size() + 8;
  dst = EncodeVarint32Ptr(dst, static_cast<uint32_t>(klen));
  memcpy(dst, user_key.data(), user_key.size());
  EncodeFixed64(dst + user_key.size(), tag);
  dst += klen;
  dst = EncodeVarint32Ptr(dst, static_cast<uint32_t>(value.size()));
  memcpy(dst, value.data(), value.size());
  return dst + value.size();
}

// Decode the inverse of WriteEncodedEntry — extract user_key, tag, value
// from an Allocate()'d buffer that the caller filled.
inline void DecodeEntry(const char* p, Slice* user_key, uint64_t* tag,
                        Slice* value) {
  uint32_t klen = 0;
  p = GetVarint32Ptr(p, p + 5, &klen);
  assert(klen >= 8);
  *user_key = Slice(p, klen - 8);
  *tag = DecodeFixed64(p + klen - 8);
  p += klen;
  uint32_t vlen = 0;
  p = GetVarint32Ptr(p, p + 5, &vlen);
  *value = Slice(p, vlen);
}

}  // anonymous namespace

struct CSPPMemTabFactory;

struct CSPPMemTab : public MemTableRep {
  static constexpr size_t Align = MainPatricia::AlignSize;
  static_assert(Align == 4, "CSPPMemTab depends on 4-byte trie alignment");

  // Per-user-key tag→value vector stored inside the trie's memory pool.
#pragma pack(push, 4)
  struct Entry {
    uint64_t tag;
    uint32_t pos;  // offset into trie mempool of length-prefixed value, or 0
    operator uint64_t() const noexcept { return tag; }  // for *_bound_0
    Slice GetValue(const void* mempool) const noexcept {
      if (size_t p = pos) {
        auto enc = static_cast<const char*>(mempool) + p * Align;
        return GetLengthPrefixedSlice(enc);
      }
      return Slice();
    }
  };
  struct VecPin {  // allocated once per user-key, may grow (COW)
    uint32_t num;  // top bit is LOCK_FLAG; bottom bits = entry count
    uint32_t pos;  // offset into trie mempool of Entry[]
  };
#pragma pack(pop)

  static size_t EncValueLen(size_t raw_val_len) {
    return raw_val_len
               ? pow2_align_up(VarintLength(raw_val_len) + raw_val_len, Align)
               : 0;
  }

  mutable MainPatricia m_trie;
  bool m_rev;
  bool m_read_by_writer_token;
  bool m_token_use_idle;
  std::atomic<bool> m_is_empty{true};
  std::atomic<bool> m_is_readonly{false};
  Logger* m_log;
  WriteBufferManager* m_wbm;          // non-owning; from factory
  std::atomic<size_t> m_reported;     // bytes currently reported to m_wbm

  CSPPMemTab(intptr_t cap, bool rev, Logger* log, bool read_by_writer_token,
             WriteBufferManager* wbm);
  ~CSPPMemTab() override;

  // Forward trie mempool growth to the WriteBufferManager so it can enforce
  // a slot-wide memory budget and (if cost_to_cache is enabled) reserve
  // dummy cache entries on the bound block cache.
  void ReconcileMem() {
    if (!m_wbm) return;
    size_t now = m_trie.mem_size_inline();
    size_t prev = m_reported.load(std::memory_order_relaxed);
    // CAS loop: only the thread that wins the CAS forwards the delta.
    // Other writers will re-read the now-bumped m_reported on their next
    // call and forward their own slice. This is correct even with many
    // concurrent inserts because WBM->ReserveMem is itself atomic.
    while (now > prev) {
      if (m_reported.compare_exchange_weak(prev, now,
              std::memory_order_relaxed,
              std::memory_order_relaxed)) {
        m_wbm->ReserveMem(now - prev);
        return;
      }
      // prev was updated by another thread; loop with the new prev.
    }
    // now <= prev: trie shrank (rare for patricia) or stale read; skip.
  }

  // Encodes the tag+val into the trie when a brand-new key is inserted.
  struct Token : public Patricia::WriterToken {
    uint64_t tag_ = UINT64_MAX;
    Slice val_;
    bool init_value(void* trie_valptr, size_t trie_valsize) noexcept final;
    void destroy_value(void* trie_valptr, size_t trie_valsize) noexcept final;
  };

  bool insert_kv(fstring user_key, Token* tok);
  bool insert_for_dup_user_key(Token* tok);

  Patricia::TokenBase* reader_token() const {
    return m_read_by_writer_token
               ? static_cast<Patricia::TokenBase*>(
                     m_trie.tls_writer_token_nn<Token>())
               : static_cast<Patricia::TokenBase*>(m_trie.tls_reader_token());
  }

  // --- vanilla MemTableRep ABI ---
  KeyHandle Allocate(const size_t len, char** buf) override;
  void Insert(KeyHandle handle) override;
  bool InsertKey(KeyHandle handle) override;
  void InsertConcurrently(KeyHandle handle) override;
  bool InsertKeyConcurrently(KeyHandle handle) override;
  bool Contains(const char* key) const override;
  void Get(const LookupKey& k, void* callback_args,
           bool (*callback_func)(void*, const char*)) override;
  void MarkReadOnly() override;
  void MarkFlushed() override;
  size_t ApproximateMemoryUsage() override;
  uint64_t ApproximateNumEntries(const Slice& start_ikey,
                                 const Slice& end_ikey) override;
  Iterator* GetIterator(Arena* arena) override;

  struct Iter;
};

CSPPMemTab::CSPPMemTab(intptr_t cap, bool rev, Logger* log,
                       bool read_by_writer_token, WriteBufferManager* wbm)
    : MemTableRep(nullptr),  // we do not use the ConcurrentArena
      m_trie(sizeof(uint32_t), cap, Patricia::MultiWriteMultiRead),
      m_rev(rev),
      m_read_by_writer_token(read_by_writer_token),
      m_token_use_idle(true),
      m_log(log),
      m_wbm(wbm),
      m_reported(0) {
  // trie starts writable; MarkReadOnly() flips it later.
}

CSPPMemTab::~CSPPMemTab() {
  if (m_wbm) {
    size_t r = m_reported.exchange(0, std::memory_order_relaxed);
    if (r > 0) m_wbm->FreeMem(r);
  }
}

// init_value runs inside the trie's insert when a *new* user_key is
// established. It allocates the VecPin + first Entry + value buffer inside
// the trie's mempool. The caller's `val_` Slice points to the buffer the
// MemTable/MemTableAllocator owns — we copy out of it before returning.
bool CSPPMemTab::Token::init_value(void* trie_valptr, size_t valsize) noexcept {
  assert(valsize == sizeof(uint32_t));
  (void)valsize;
  auto trie = static_cast<MainPatricia*>(m_trie);
  const size_t enc_val_len = CSPPMemTab::EncValueLen(val_.size());
  const size_t vec_pin_pos =
      trie->mem_alloc(sizeof(VecPin) + sizeof(Entry) + enc_val_len);
  if (vec_pin_pos == MainPatricia::mem_alloc_fail) {
    return false;
  }
  const size_t entry_pos = vec_pin_pos + (sizeof(VecPin) / Align);
  auto vec_pin = reinterpret_cast<VecPin*>(trie->mem_get(vec_pin_pos));
  auto entry = reinterpret_cast<Entry*>(vec_pin + 1);
  *reinterpret_cast<uint32_t*>(trie_valptr) =
      static_cast<uint32_t>(vec_pin_pos);
  vec_pin->pos = static_cast<uint32_t>(entry_pos);
  vec_pin->num = 1;
  entry->tag = tag_;
  if (val_.size() > 0) {
    auto enc_val_ptr = reinterpret_cast<byte_t*>(entry + 1);
    const size_t enc_val_pos =
        vec_pin_pos + ((sizeof(VecPin) + sizeof(Entry)) / Align);
    entry->pos = static_cast<uint32_t>(enc_val_pos);
    char* p = EncodeVarint32(reinterpret_cast<char*>(enc_val_ptr),
                             static_cast<uint32_t>(val_.size()));
    memcpy(p, val_.data(), val_.size());
  } else {
    entry->pos = 0;
  }
  return true;
}

// Called when a concurrent inserter for the same user_key already won and we
// need to roll back our pre-allocation.
void CSPPMemTab::Token::destroy_value(void* trie_valptr,
                                      size_t valsize) noexcept {
  assert(valsize == sizeof(uint32_t));
  (void)valsize;
  auto trie = static_cast<MainPatricia*>(m_trie);
  const uint32_t vec_pin_pos = *reinterpret_cast<const uint32_t*>(trie_valptr);
  const size_t enc_val_len = CSPPMemTab::EncValueLen(val_.size());
  trie->mem_free(vec_pin_pos, sizeof(VecPin) + sizeof(Entry) + enc_val_len);
}

bool CSPPMemTab::insert_kv(fstring user_key, Token* tok) {
  uint32_t value_storage = UINT32_MAX;
  if (LIKELY(m_trie.insert(user_key, &value_storage, tok))) {
    return tok->has_value();
  }
  return insert_for_dup_user_key(tok);
}

// Duplicate user_key: append a new (tag,value) entry to the VecPin's Entry[]
// (copy-on-write into a larger Entry[] when capacity is exhausted).
bool CSPPMemTab::insert_for_dup_user_key(Token* tok) {
  auto trie = &m_trie;
  const uint32_t vec_pin_pos = trie->value_of<uint32_t>(*tok);
  auto vec_pin = reinterpret_cast<VecPin*>(trie->mem_get(vec_pin_pos));
  uint32_t num;
  while (LOCK_FLAG &
         (num = as_atomic(vec_pin->num)
                    .fetch_or(LOCK_FLAG, std::memory_order_acquire))) {
    std::this_thread::yield();
  }
  const uint32_t old_cap =
      (num & (num - 1)) == 0 ? num : 2u << terark_bsr_u32(num);
  const uint32_t entry_old_pos = vec_pin->pos;
  auto entry_old = reinterpret_cast<Entry*>(trie->mem_get(entry_old_pos));
  const uint64_t curr_seq = tok->tag_ >> 8;
  const uint64_t last_seq = entry_old[num - 1].tag >> 8;
  if (UNLIKELY(curr_seq == last_seq)) {
    as_atomic(vec_pin->num).store(num, std::memory_order_release);
    return false;  // duplicate (user_key, seq) — caller bug
  }
  trie->mem_gc(tok);
  size_t enc_val_pos;
  if (tok->val_.size() > 0) {
    enc_val_pos =
        trie->mem_alloc(VarintLength(tok->val_.size()) + tok->val_.size());
    if (enc_val_pos == MainPatricia::mem_alloc_fail) {
      as_atomic(vec_pin->num).store(num, std::memory_order_release);
      return false;
    }
    char* p = EncodeVarint32(static_cast<char*>(trie->mem_get(enc_val_pos)),
                             static_cast<uint32_t>(tok->val_.size()));
    memcpy(p, tok->val_.data(), tok->val_.size());
  } else {
    enc_val_pos = 0;
  }
  if (num < old_cap && last_seq < curr_seq) {
    entry_old[num].pos = static_cast<uint32_t>(enc_val_pos);
    entry_old[num].tag = tok->tag_;
    as_atomic(vec_pin->num).store(num + 1, std::memory_order_release);
    return true;
  }
  const uint32_t new_cap = num == old_cap ? old_cap * 2 : old_cap;
  const size_t entry_cow_pos = trie->mem_alloc(sizeof(Entry) * new_cap);
  if (entry_cow_pos == MainPatricia::mem_alloc_fail) {
    as_atomic(vec_pin->num).store(num, std::memory_order_release);
    return false;
  }
  auto entry_cow = reinterpret_cast<Entry*>(trie->mem_get(entry_cow_pos));
  if (LIKELY(last_seq < curr_seq)) {
    memcpy(entry_cow, entry_old, sizeof(Entry) * num);
    entry_cow[num].pos = static_cast<uint32_t>(enc_val_pos);
    entry_cow[num].tag = tok->tag_;
  } else {
    auto idx = lower_bound_0(entry_old, num, curr_seq << 8);
    if (UNLIKELY(entry_old[idx].tag >> 8 == curr_seq)) {
      as_atomic(vec_pin->num).store(num, std::memory_order_release);
      trie->mem_free(entry_cow_pos, sizeof(Entry) * new_cap);
      if (enc_val_pos) {
        trie->mem_free(enc_val_pos,
                       VarintLength(tok->val_.size()) + tok->val_.size());
      }
      return false;
    }
    memcpy(entry_cow, entry_old, sizeof(Entry) * idx);
    entry_cow[idx].pos = static_cast<uint32_t>(enc_val_pos);
    entry_cow[idx].tag = tok->tag_;
    memcpy(entry_cow + idx + 1, entry_old + idx,
           sizeof(Entry) * (num - idx));
  }
  vec_pin->pos = static_cast<uint32_t>(entry_cow_pos);
  as_atomic(vec_pin->num).store(num + 1, std::memory_order_release);
  trie->mem_lazy_free(entry_old_pos, sizeof(Entry) * old_cap, tok);
  return true;
}

// --- vanilla MemTableRep ABI implementations ---

KeyHandle CSPPMemTab::Allocate(const size_t len, char** buf) {
  // The buffer's lifetime is until Insert() returns. We use the heap; the
  // alloc/free cost per insert is negligible vs. trie work.
  *buf = new char[len];
  return static_cast<KeyHandle>(*buf);
}

void CSPPMemTab::Insert(KeyHandle handle) { (void)InsertKey(handle); }
void CSPPMemTab::InsertConcurrently(KeyHandle handle) {
  (void)InsertKeyConcurrently(handle);
}

bool CSPPMemTab::InsertKey(KeyHandle handle) {
  return InsertKeyConcurrently(handle);
}

bool CSPPMemTab::InsertKeyConcurrently(KeyHandle handle) {
  Slice user_key, value;
  uint64_t tag = 0;
  char* buf = static_cast<char*>(handle);
  DecodeEntry(buf, &user_key, &tag, &value);

  m_is_empty.store(false, std::memory_order_relaxed);
  Token* token = m_trie.tls_writer_token_nn<Token>();
  token->acquire(&m_trie);
  token->tag_ = tag;
  token->val_ = value;
  bool ok = insert_kv(fstring(user_key.data(), user_key.size()), token);
  m_token_use_idle ? token->idle() : token->release();

  delete[] buf;
  if (ok) {
    // Forward trie mempool growth to the WriteBufferManager (if configured).
    ReconcileMem();
  }
  return ok;
}

bool CSPPMemTab::Contains(const char* key) const {
  // `key` is a length-prefixed internal_key (user_key + 8-byte tag).
  Slice internal_key = GetLengthPrefixedSlice(key);
  if (internal_key.size() < 8) return false;
  fstring user_key(internal_key.data(), internal_key.size() - 8);
  const uint64_t find_tag =
      DecodeFixed64(internal_key.data() + internal_key.size() - 8);
  auto token = reader_token();
  token->acquire(&m_trie);
  if (!m_trie.lookup(user_key, token)) {
    m_token_use_idle ? token->idle() : token->release();
    return false;
  }
  auto vec_pin = reinterpret_cast<VecPin*>(
      m_trie.mem_get(m_trie.value_of<uint32_t>(*token)));
  const uint32_t num = vec_pin->num & ~LOCK_FLAG;
  auto entry = reinterpret_cast<Entry*>(m_trie.mem_get(vec_pin->pos));
  const bool found = binary_search_0(entry, num, find_tag);
  m_token_use_idle ? token->idle() : token->release();
  return found;
}

void CSPPMemTab::Get(const LookupKey& k, void* callback_args,
                     bool (*callback_func)(void*, const char*)) {
  if (UNLIKELY(m_is_empty.load(std::memory_order_relaxed))) return;
  Slice ikey = k.internal_key();
  if (ikey.size() < 8) return;
  fstring user_key(ikey.data(), ikey.size() - 8);
  const uint64_t find_tag = DecodeFixed64(ikey.data() + ikey.size() - 8);

  auto token = reader_token();
  token->acquire(&m_trie);
  if (!m_trie.lookup(user_key, token)) {
    m_token_use_idle ? token->idle() : token->release();
    return;
  }
  auto vec_pin = reinterpret_cast<VecPin*>(
      m_trie.mem_get(m_trie.value_of<uint32_t>(*token)));
  const uint32_t num = vec_pin->num & ~LOCK_FLAG;
  auto entry = reinterpret_cast<Entry*>(m_trie.mem_get(vec_pin->pos));
  intptr_t idx = upper_bound_0(entry, num, find_tag);

  std::string scratch;  // reused across callback invocations
  while (idx-- > 0) {
    Slice val = entry[idx].GetValue(m_trie.mem_get(0));
    scratch.resize(EncodedEntryLen(user_key.size(), val.size()));
    WriteEncodedEntry(&scratch[0],
                      Slice(user_key.data(), user_key.size()),
                      entry[idx].tag, val);
    if (!callback_func(callback_args, scratch.data())) break;
  }
  m_token_use_idle ? token->idle() : token->release();
}

void CSPPMemTab::MarkReadOnly() {
  m_is_readonly.store(true, std::memory_order_release);
  m_trie.set_readonly();  // terark's set_readonly takes no args
}

void CSPPMemTab::MarkFlushed() { /* nothing extra to do */ }

size_t CSPPMemTab::ApproximateMemoryUsage() {
  return m_trie.mem_size_inline();
}

uint64_t CSPPMemTab::ApproximateNumEntries(const Slice&, const Slice&) {
  // Coarse approximation — the trie itself does not maintain a per-range
  // count. Returning 0 is a documented safe default per the base class.
  return 0;
}

// --- Iterator over the trie, exposing the vanilla MemTableRep::Iterator ABI ---
struct CSPPMemTab::Iter : public MemTableRep::Iterator {
  CSPPMemTab* m_tab;
  Patricia::Iterator* m_iter = nullptr;
  int m_idx = -1;
  bool m_rev;
  std::string m_key_buf;  // owns memory backing key() return value

  explicit Iter(CSPPMemTab* t) : m_tab(t), m_rev(t->m_rev) {}
  ~Iter() override {
    if (m_iter) m_iter->dispose();
  }

  bool LazyInit() {
    if (LIKELY(m_iter != nullptr)) return true;
    if (m_tab->m_is_empty.load(std::memory_order_relaxed)) return false;
    m_iter = m_tab->m_trie.new_iter();
    return true;
  }

  // After positioning m_iter at a user_key, set m_idx to a specific Entry
  // and rebuild m_key_buf for key() to return.
  void EmitKey(int idx) {
    m_idx = idx;
    auto* mempool = static_cast<const char*>(m_tab->m_trie.mem_get(0));
    const uint32_t vec_pin_pos =
        *reinterpret_cast<const uint32_t*>(mempool + m_iter->get_valpos());
    auto vec_pin = reinterpret_cast<const VecPin*>(mempool + Align * vec_pin_pos);
    auto entry =
        reinterpret_cast<const Entry*>(mempool + Align * vec_pin->pos);
    const uint64_t tag = entry[idx].tag;
    const Slice val = entry[idx].GetValue(mempool);
    fstring uk = m_iter->word();
    m_key_buf.resize(
        EncodedEntryLen(uk.size(), val.size()));
    WriteEncodedEntry(&m_key_buf[0],
                      Slice(uk.data(), uk.size()), tag, val);
  }

  int EntryCount() const {
    auto* mempool = static_cast<const char*>(m_tab->m_trie.mem_get(0));
    const uint32_t vec_pin_pos =
        *reinterpret_cast<const uint32_t*>(mempool + m_iter->get_valpos());
    auto vec_pin = reinterpret_cast<const VecPin*>(mempool + Align * vec_pin_pos);
    return static_cast<int>(vec_pin->num & ~LOCK_FLAG);
  }

  bool Valid() const override { return m_idx >= 0; }

  const char* key() const override {
    assert(m_idx >= 0);
    return m_key_buf.data();
  }

  void Next() override {
    assert(m_idx >= 0);
    if (m_idx-- == 0) {
      const bool ok = m_rev ? m_iter->decr() : m_iter->incr();
      if (!ok) { m_idx = -1; return; }
      EmitKey(EntryCount() - 1);
    } else {
      EmitKey(m_idx);
    }
  }

  void Prev() override {
    assert(m_idx >= 0);
    const int n = EntryCount();
    if (++m_idx == n) {
      const bool ok = m_rev ? m_iter->incr() : m_iter->decr();
      if (!ok) { m_idx = -1; return; }
      EmitKey(0);
    } else {
      EmitKey(m_idx);
    }
  }

  void Seek(const Slice& ikey, const char*) override {
    if (!LazyInit()) { m_idx = -1; return; }
    if (ikey.size() < 8) { m_idx = -1; return; }
    fstring uk(ikey.data(), ikey.size() - 8);
    const uint64_t find_tag = DecodeFixed64(ikey.data() + ikey.size() - 8);
    const bool ok =
        m_rev ? m_iter->seek_rev_lower_bound(uk) : m_iter->seek_lower_bound(uk);
    if (!ok) { m_idx = -1; return; }
    const int n = EntryCount();
    auto* mempool = static_cast<const char*>(m_tab->m_trie.mem_get(0));
    const uint32_t vec_pin_pos =
        *reinterpret_cast<const uint32_t*>(mempool + m_iter->get_valpos());
    auto vec_pin = reinterpret_cast<const VecPin*>(mempool + Align * vec_pin_pos);
    auto entry =
        reinterpret_cast<const Entry*>(mempool + Align * vec_pin->pos);
    if (m_iter->word() == uk) {
      int idx = static_cast<int>(upper_bound_0(entry, n, find_tag)) - 1;
      if (idx >= 0) { EmitKey(idx); return; }
      const bool more = m_rev ? m_iter->decr() : m_iter->incr();
      if (!more) { m_idx = -1; return; }
      EmitKey(EntryCount() - 1);
      return;
    }
    EmitKey(n - 1);
  }

  void SeekForPrev(const Slice& ikey, const char*) override {
    if (!LazyInit()) { m_idx = -1; return; }
    if (ikey.size() < 8) { m_idx = -1; return; }
    fstring uk(ikey.data(), ikey.size() - 8);
    const uint64_t find_tag = DecodeFixed64(ikey.data() + ikey.size() - 8);
    const bool ok =
        m_rev ? m_iter->seek_lower_bound(uk) : m_iter->seek_rev_lower_bound(uk);
    if (!ok) { m_idx = -1; return; }
    const int n = EntryCount();
    auto* mempool = static_cast<const char*>(m_tab->m_trie.mem_get(0));
    const uint32_t vec_pin_pos =
        *reinterpret_cast<const uint32_t*>(mempool + m_iter->get_valpos());
    auto vec_pin = reinterpret_cast<const VecPin*>(mempool + Align * vec_pin_pos);
    auto entry =
        reinterpret_cast<const Entry*>(mempool + Align * vec_pin->pos);
    if (m_iter->word() == uk) {
      const int idx = static_cast<int>(lower_bound_0(entry, n, find_tag));
      if (idx != n) { EmitKey(idx); return; }
      const bool more = m_rev ? m_iter->incr() : m_iter->decr();
      if (!more) { m_idx = -1; return; }
      EmitKey(0);
      return;
    }
    EmitKey(0);
  }

  void SeekToFirst() override {
    if (!LazyInit()) { m_idx = -1; return; }
    const bool ok = m_rev ? m_iter->seek_end() : m_iter->seek_begin();
    if (!ok) { m_idx = -1; return; }
    EmitKey(EntryCount() - 1);
  }

  void SeekToLast() override {
    if (!LazyInit()) { m_idx = -1; return; }
    const bool ok = m_rev ? m_iter->seek_begin() : m_iter->seek_end();
    if (!ok) { m_idx = -1; return; }
    EmitKey(0);
  }
};

MemTableRep::Iterator* CSPPMemTab::GetIterator(Arena* arena) {
  if (arena) {
    void* mem = arena->AllocateAligned(sizeof(Iter));
    return new (mem) Iter(this);
  }
  return new Iter(this);
}

// --- Factory ---

struct CSPPMemTabFactory final : public MemTableRepFactory {
  size_t mem_cap_;
  bool read_by_writer_token_;
  WriteBufferManager* write_buffer_manager_;
  CSPPMemTabFactory(size_t mem_cap, bool read_by_writer_token,
                    WriteBufferManager* wbm)
      : mem_cap_(mem_cap),
        read_by_writer_token_(read_by_writer_token),
        write_buffer_manager_(wbm) {}
  const char* Name() const override { return "CSPPMemTab"; }
  using MemTableRepFactory::CreateMemTableRep;
  MemTableRep* CreateMemTableRep(const MemTableRep::KeyComparator& /*cmp*/,
                                 Allocator* /*allocator*/,
                                 const SliceTransform* /*transform*/,
                                 Logger* logger) override {
    // Vanilla MemTableRep::KeyComparator doesn't expose the underlying
    // user comparator, so we cannot detect reverse comparators here. CSPP
    // sorts keys byte-wise lexicographically, which matches the standard
    // bytewise comparator used by Flink statebackend and most workloads.
    intptr_t cap = mem_cap_ > 0 ? static_cast<intptr_t>(mem_cap_)
                                : intptr_t(2) << 30;  // 2 GiB default
    return new CSPPMemTab(cap, /*rev=*/false, logger, read_by_writer_token_,
                          write_buffer_manager_);
  }
  bool IsInsertConcurrentlySupported() const override { return true; }
  bool CanHandleDuplicatedKey() const override { return true; }
};

MemTableRepFactory* NewCSPPMemTableRepFactory(
    size_t mem_cap, bool read_by_writer_token,
    WriteBufferManager* write_buffer_manager) {
  return new CSPPMemTabFactory(mem_cap, read_by_writer_token,
                               write_buffer_manager);
}

}  // namespace ROCKSDB_NAMESPACE
