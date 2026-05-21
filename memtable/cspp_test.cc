// SPDX-License-Identifier: GPL-2.0 OR Apache-2.0
// Unit tests for the CSPP memtable rep, exercising the MemTableRep ABI
// directly (Allocate / Insert / Contains / Iterator / MarkReadOnly /
// ApproximateMemoryUsage). The WriteBufferManager integration has its own
// test (cspp_wbm_test.cc); this file deliberately stays away from
// WBM so the surface under test is just the memtable itself.
//
// Build: make WITH_CSPP_MEMTABLE=1 cspp_test
// Run:   ./cspp_test

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "db/dbformat.h"
#include "memory/arena.h"
#include "port/stack_trace.h"
#include "rocksdb/comparator.h"
#include "rocksdb/memtablerep.h"
#include "rocksdb/slice.h"
#include "test_util/testharness.h"
#include "util/coding.h"
#include "util/hash.h"
#include "util/random.h"

namespace ROCKSDB_NAMESPACE {

namespace {

// MemTableRep::KeyComparator backed by an InternalKeyComparator. CSPP itself
// ignores the comparator (it always sorts byte-wise), but CreateMemTableRep
// requires one.
struct InternalKeyComparatorAdapter : public MemTableRep::KeyComparator {
  const InternalKeyComparator* cmp;
  explicit InternalKeyComparatorAdapter(const InternalKeyComparator* c)
      : cmp(c) {}
  int operator()(const char* a, const char* b) const override {
    return cmp->Compare(GetLengthPrefixedSlice(a), GetLengthPrefixedSlice(b));
  }
  int operator()(const char* a, const Slice& kb) const override {
    return cmp->Compare(GetLengthPrefixedSlice(a), kb);
  }
};

struct RepBundle {
  std::unique_ptr<MemTableRepFactory> factory;
  std::unique_ptr<Arena> arena;
  std::unique_ptr<InternalKeyComparator> ikc;
  std::unique_ptr<InternalKeyComparatorAdapter> cmp;
  std::unique_ptr<MemTableRep> rep;
};

std::unique_ptr<RepBundle> MakeRep() {
  auto bundle = std::make_unique<RepBundle>();
  // 32 MiB initial capacity; plenty for unit tests.
  bundle->factory.reset(NewCSPPMemTableRepFactory(/*mem_cap=*/32ull << 20));
  EXPECT_NE(bundle->factory, nullptr);
  bundle->arena = std::make_unique<Arena>();
  bundle->ikc = std::make_unique<InternalKeyComparator>(BytewiseComparator());
  bundle->cmp =
      std::make_unique<InternalKeyComparatorAdapter>(bundle->ikc.get());
  bundle->rep.reset(bundle->factory->CreateMemTableRep(
      *bundle->cmp, bundle->arena.get(), /*transform=*/nullptr,
      /*logger=*/nullptr));
  EXPECT_NE(bundle->rep, nullptr);
  return bundle;
}

// Build the length-prefixed encoded entry that MemTableRep::Insert expects
// (varint(ikey_size) | ikey | varint(vlen) | value), allocate it via the rep
// and call Insert.
void InsertOne(MemTableRep* rep, const std::string& user_key, uint64_t seq,
               ValueType type, const Slice& value) {
  const size_t internal_key_size = user_key.size() + 8;
  const size_t encoded_len = VarintLength(internal_key_size) +
                             internal_key_size + VarintLength(value.size()) +
                             value.size();
  char* buf = nullptr;
  KeyHandle handle = rep->Allocate(encoded_len, &buf);
  ASSERT_NE(buf, nullptr);
  char* p = EncodeVarint32(buf, static_cast<uint32_t>(internal_key_size));
  memcpy(p, user_key.data(), user_key.size());
  p += user_key.size();
  EncodeFixed64(p, PackSequenceAndType(seq, type));
  p += 8;
  p = EncodeVarint32(p, static_cast<uint32_t>(value.size()));
  memcpy(p, value.data(), value.size());
  rep->Insert(handle);
}

void InsertValue(MemTableRep* rep, const std::string& user_key, uint64_t seq,
                 const Slice& value) {
  InsertOne(rep, user_key, seq, kTypeValue, value);
}

// Build a length-prefixed internal_key for use with MemTableRep::Contains.
std::string MakeLPInternalKey(const std::string& user_key, uint64_t seq,
                              ValueType type) {
  std::string buf;
  const size_t ikey_size = user_key.size() + 8;
  PutVarint32(&buf, static_cast<uint32_t>(ikey_size));
  buf.append(user_key);
  char tag[8];
  EncodeFixed64(tag, PackSequenceAndType(seq, type));
  buf.append(tag, 8);
  return buf;
}

// Returns the user_key extracted from a MemTableRep::Iterator::key().
std::string IterUserKey(const char* key_ptr) {
  Slice ikey = GetLengthPrefixedSlice(key_ptr);
  // Strip the trailing 8-byte tag.
  return std::string(ikey.data(), ikey.size() - 8);
}

ParsedInternalKey IterParsedKey(const char* key_ptr) {
  Slice ikey = GetLengthPrefixedSlice(key_ptr);
  ParsedInternalKey parsed;
  EXPECT_TRUE(ParseInternalKey(ikey, &parsed, /*log_err_key=*/false).ok());
  return parsed;
}

}  // namespace

class CSPPMemTabTest : public testing::Test {};

TEST_F(CSPPMemTabTest, EmptyRep) {
  auto bundle = MakeRep();
  EXPECT_EQ(bundle->rep->ApproximateNumEntries(Slice(), Slice()), 0u);

  std::unique_ptr<MemTableRep::Iterator> it(bundle->rep->GetIterator(nullptr));
  it->SeekToFirst();
  EXPECT_FALSE(it->Valid());
  it->SeekToLast();
  EXPECT_FALSE(it->Valid());

  std::string lp = MakeLPInternalKey("absent", 1, kTypeValue);
  EXPECT_FALSE(bundle->rep->Contains(lp.data()));
}

TEST_F(CSPPMemTabTest, InsertAndContains) {
  auto bundle = MakeRep();
  const std::vector<std::string> keys = {"alpha", "bravo",   "charlie",
                                         "delta", "echo",    "foxtrot",
                                         "golf",  "hotel",   "india",
                                         "juliett"};
  std::vector<uint64_t> seqs;
  uint64_t seq = 1;
  for (const auto& k : keys) {
    seqs.push_back(seq);
    InsertValue(bundle->rep.get(), k, seq++, Slice("v_" + k));
  }
  // Contains matches on (user_key, tag) exactly — must look up with the same
  // seq that was used at insert time.
  for (size_t i = 0; i < keys.size(); i++) {
    std::string lp = MakeLPInternalKey(keys[i], seqs[i], kTypeValue);
    EXPECT_TRUE(bundle->rep->Contains(lp.data())) << "missing " << keys[i];
  }
  std::string lp_absent = MakeLPInternalKey("zulu", 1, kTypeValue);
  EXPECT_FALSE(bundle->rep->Contains(lp_absent.data()));
}

TEST_F(CSPPMemTabTest, IteratorYieldsKeysInSortedOrder) {
  auto bundle = MakeRep();
  // Insert in non-sorted order; iterator must return them ascending.
  const std::vector<std::string> order = {"delta",   "alpha",  "foxtrot",
                                          "charlie", "bravo",  "echo"};
  uint64_t seq = 1;
  for (const auto& k : order) {
    InsertValue(bundle->rep.get(), k, seq++, Slice("v"));
  }
  std::unique_ptr<MemTableRep::Iterator> it(bundle->rep->GetIterator(nullptr));
  std::vector<std::string> seen;
  for (it->SeekToFirst(); it->Valid(); it->Next()) {
    seen.push_back(IterUserKey(it->key()));
  }
  std::vector<std::string> expected = order;
  std::sort(expected.begin(), expected.end());
  EXPECT_EQ(seen, expected);
}

TEST_F(CSPPMemTabTest, ReverseIteration) {
  auto bundle = MakeRep();
  const std::vector<std::string> order = {"alpha", "bravo", "charlie",
                                          "delta", "echo"};
  uint64_t seq = 1;
  for (const auto& k : order) {
    InsertValue(bundle->rep.get(), k, seq++, Slice("v"));
  }
  std::unique_ptr<MemTableRep::Iterator> it(bundle->rep->GetIterator(nullptr));
  std::vector<std::string> seen;
  for (it->SeekToLast(); it->Valid(); it->Prev()) {
    seen.push_back(IterUserKey(it->key()));
  }
  std::vector<std::string> expected = order;
  std::sort(expected.begin(), expected.end(), std::greater<std::string>());
  EXPECT_EQ(seen, expected);
}

TEST_F(CSPPMemTabTest, SeekEdgeCases) {
  auto bundle = MakeRep();
  InsertValue(bundle->rep.get(), "c", 1, Slice("v"));
  InsertValue(bundle->rep.get(), "e", 2, Slice("v"));
  InsertValue(bundle->rep.get(), "g", 3, Slice("v"));

  std::unique_ptr<MemTableRep::Iterator> it(bundle->rep->GetIterator(nullptr));

  auto seek = [&](const std::string& target) {
    // Seek to (target, kMaxSeq) lands on the first internal_key with
    // user_key >= target, since seq sorts descending inside a user_key.
    std::string lp = MakeLPInternalKey(target, kMaxSequenceNumber, kTypeValue);
    Slice ikey = GetLengthPrefixedSlice(lp.data());
    it->Seek(ikey, lp.data());
  };
  auto seek_for_prev = [&](const std::string& target) {
    // SeekForPrev to (target, 0) lands on the largest internal_key with
    // user_key <= target.
    std::string lp = MakeLPInternalKey(target, 0, kTypeValue);
    Slice ikey = GetLengthPrefixedSlice(lp.data());
    it->SeekForPrev(ikey, lp.data());
  };

  // Seek: first key >= target
  seek("a");
  ASSERT_TRUE(it->Valid());
  EXPECT_EQ(IterUserKey(it->key()), "c");
  seek("c");
  ASSERT_TRUE(it->Valid());
  EXPECT_EQ(IterUserKey(it->key()), "c");
  seek("d");
  ASSERT_TRUE(it->Valid());
  EXPECT_EQ(IterUserKey(it->key()), "e");
  seek("z");
  EXPECT_FALSE(it->Valid());

  // SeekForPrev: largest key <= target
  seek_for_prev("a");
  EXPECT_FALSE(it->Valid());
  seek_for_prev("c");
  ASSERT_TRUE(it->Valid());
  EXPECT_EQ(IterUserKey(it->key()), "c");
  seek_for_prev("d");
  ASSERT_TRUE(it->Valid());
  EXPECT_EQ(IterUserKey(it->key()), "c");
  seek_for_prev("z");
  ASSERT_TRUE(it->Valid());
  EXPECT_EQ(IterUserKey(it->key()), "g");
}

TEST_F(CSPPMemTabTest, OverwritePreservesAllVersions) {
  auto bundle = MakeRep();
  // Three writes to the same user_key with increasing seq numbers.
  InsertValue(bundle->rep.get(), "key", 10, Slice("v1"));
  InsertValue(bundle->rep.get(), "key", 20, Slice("v2"));
  InsertValue(bundle->rep.get(), "key", 30, Slice("v3"));

  // Iterator should walk all three entries. Per RocksDB internal-key order
  // they sort by (user_key asc, seq desc), so the highest seq comes first.
  std::unique_ptr<MemTableRep::Iterator> it(bundle->rep->GetIterator(nullptr));
  std::vector<uint64_t> seqs;
  for (it->SeekToFirst(); it->Valid(); it->Next()) {
    auto parsed = IterParsedKey(it->key());
    EXPECT_EQ(parsed.user_key.ToString(), "key");
    seqs.push_back(parsed.sequence);
  }
  ASSERT_EQ(seqs.size(), 3u);
  EXPECT_EQ(seqs[0], 30u);
  EXPECT_EQ(seqs[1], 20u);
  EXPECT_EQ(seqs[2], 10u);
}

TEST_F(CSPPMemTabTest, DeletionTombstoneIsVisible) {
  auto bundle = MakeRep();
  InsertValue(bundle->rep.get(), "key", 1, Slice("v"));
  InsertOne(bundle->rep.get(), "key", 2, kTypeDeletion, Slice());

  std::unique_ptr<MemTableRep::Iterator> it(bundle->rep->GetIterator(nullptr));
  std::vector<ValueType> types;
  for (it->SeekToFirst(); it->Valid(); it->Next()) {
    types.push_back(IterParsedKey(it->key()).type);
  }
  ASSERT_EQ(types.size(), 2u);
  EXPECT_EQ(types[0], kTypeDeletion);  // higher seq comes first
  EXPECT_EQ(types[1], kTypeValue);
}

TEST_F(CSPPMemTabTest, MarkReadOnlyKeepsReadsWorking) {
  auto bundle = MakeRep();
  const std::vector<std::string> keys = {"a", "b", "c", "d", "e"};
  std::vector<uint64_t> seqs;
  uint64_t seq = 1;
  for (const auto& k : keys) {
    seqs.push_back(seq);
    InsertValue(bundle->rep.get(), k, seq++, Slice("v"));
  }
  bundle->rep->MarkReadOnly();

  // Contains still works (matches on exact tag).
  for (size_t i = 0; i < keys.size(); i++) {
    std::string lp = MakeLPInternalKey(keys[i], seqs[i], kTypeValue);
    EXPECT_TRUE(bundle->rep->Contains(lp.data()));
  }
  // Iterator still works.
  std::unique_ptr<MemTableRep::Iterator> it(bundle->rep->GetIterator(nullptr));
  size_t n = 0;
  for (it->SeekToFirst(); it->Valid(); it->Next()) {
    n++;
  }
  EXPECT_EQ(n, keys.size());

  // MarkFlushed must not crash even with no outstanding work.
  bundle->rep->MarkFlushed();
}

TEST_F(CSPPMemTabTest, ApproximateMemoryUsageGrowsWithInserts) {
  auto bundle = MakeRep();
  const size_t baseline = bundle->rep->ApproximateMemoryUsage();
  // Patricia pre-allocates a 2 MiB chunk and reports its mempool size,
  // which only grows when the next chunk is allocated. Write enough
  // (~5 MiB of values) to force at least one chunk extension.
  std::string value(1024, 'x');
  for (int i = 0; i < 5000; i++) {
    char keybuf[32];
    int n = snprintf(keybuf, sizeof(keybuf), "k_%08d", i);
    InsertValue(bundle->rep.get(),
                std::string(keybuf, static_cast<size_t>(n)),
                static_cast<uint64_t>(i + 1), Slice(value));
  }
  const size_t after = bundle->rep->ApproximateMemoryUsage();
  EXPECT_GT(after, baseline);
}

TEST_F(CSPPMemTabTest, ConcurrentInsertsAllVisible) {
  auto bundle = MakeRep();
  constexpr int kThreads = 4;
  constexpr int kPerThread = 1000;

  std::vector<std::thread> workers;
  std::atomic<uint64_t> seq_counter{1};
  for (int t = 0; t < kThreads; t++) {
    workers.emplace_back([&, t]() {
      for (int i = 0; i < kPerThread; i++) {
        char keybuf[32];
        int n = snprintf(keybuf, sizeof(keybuf), "t%d_k%06d", t, i);
        InsertValue(bundle->rep.get(),
                    std::string(keybuf, static_cast<size_t>(n)),
                    seq_counter.fetch_add(1), Slice("v"));
      }
    });
  }
  for (auto& w : workers) w.join();

  std::unique_ptr<MemTableRep::Iterator> it(bundle->rep->GetIterator(nullptr));
  std::set<std::string> seen;
  for (it->SeekToFirst(); it->Valid(); it->Next()) {
    seen.insert(IterUserKey(it->key()));
  }
  EXPECT_EQ(seen.size(),
            static_cast<size_t>(kThreads) * static_cast<size_t>(kPerThread));
}

// ---------------------------------------------------------------------------
// Patricia-level correctness — exercises the underlying MainPatricia data
// structure through CSPP's MemTableRep interface.

TEST_F(CSPPMemTabTest, RandomInsertAndLookupLargeN) {
  auto bundle = MakeRep();
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> len_dist(4, 32);
  std::set<std::string> inserted;
  std::vector<std::pair<std::string, uint64_t>> ordered;
  uint64_t seq = 1;

  // Insert ~5000 unique random keys with random lengths.
  while (inserted.size() < 5000) {
    int len = len_dist(rng);
    std::string k(static_cast<size_t>(len), '\0');
    for (int i = 0; i < len; i++) {
      k[i] = static_cast<char>(rng() & 0xFF);
    }
    if (inserted.insert(k).second) {
      ordered.emplace_back(k, seq);
      InsertValue(bundle->rep.get(), k, seq, Slice("v"));
      seq++;
    }
  }

  // Iterator must yield every key in sorted byte order.
  std::unique_ptr<MemTableRep::Iterator> it(bundle->rep->GetIterator(nullptr));
  std::vector<std::string> seen;
  for (it->SeekToFirst(); it->Valid(); it->Next()) {
    seen.push_back(IterUserKey(it->key()));
  }
  std::vector<std::string> expected(inserted.begin(), inserted.end());
  ASSERT_EQ(seen.size(), expected.size());
  for (size_t i = 0; i < seen.size(); i++) {
    ASSERT_EQ(seen[i], expected[i]) << "iterator diverges at index " << i;
  }

  // Random Contains checks — 200 inserted (must be true) and 200 absent
  // (must be false; skip if collision with inserted set).
  std::uniform_int_distribution<size_t> idx_dist(0, ordered.size() - 1);
  for (int i = 0; i < 200; i++) {
    const auto& [k, kseq] = ordered[idx_dist(rng)];
    std::string lp = MakeLPInternalKey(k, kseq, kTypeValue);
    EXPECT_TRUE(bundle->rep->Contains(lp.data()))
        << "missing inserted key (len=" << k.size() << ")";
  }
  int absent_checked = 0;
  while (absent_checked < 200) {
    int len = len_dist(rng);
    std::string k(static_cast<size_t>(len), '\0');
    for (int j = 0; j < len; j++) k[j] = static_cast<char>(rng() & 0xFF);
    if (inserted.count(k)) continue;
    std::string lp = MakeLPInternalKey(k, 1, kTypeValue);
    EXPECT_FALSE(bundle->rep->Contains(lp.data())) << "false positive";
    absent_checked++;
  }
}

TEST_F(CSPPMemTabTest, VariableLengthKeys) {
  // Patricia branching is shape-dependent; mix widely varying key lengths
  // to exercise both short and long prefixes.
  auto bundle = MakeRep();
  std::vector<std::string> keys;
  uint64_t seq = 1;
  for (int len : {1, 16, 128, 1024}) {
    for (int i = 0; i < 5; i++) {
      std::string k(static_cast<size_t>(len), '\0');
      for (int j = 0; j < len; j++) {
        k[j] = static_cast<char>((len * 31 + i * 17 + j) & 0xFF);
      }
      keys.push_back(k);
      InsertValue(bundle->rep.get(), k, seq++, Slice("v"));
    }
  }
  std::unique_ptr<MemTableRep::Iterator> it(bundle->rep->GetIterator(nullptr));
  std::vector<std::string> seen;
  for (it->SeekToFirst(); it->Valid(); it->Next()) {
    seen.push_back(IterUserKey(it->key()));
  }
  std::vector<std::string> sorted = keys;
  std::sort(sorted.begin(), sorted.end());
  EXPECT_EQ(seen, sorted);
}

// ---------------------------------------------------------------------------
// Concurrent insert + concurrent read scaffolding, modelled on
// memtable/inlineskiplist_test.cc's ConcurrentTest. The data structure
// underlying CSPP (MainPatricia) advertises lock-free concurrent inserts
// and concurrent reads against ongoing writes — this exercises both.
//
// Invariants checked by ReadStep:
//   (1) iteration is sorted: user_key ascending, and within a user_key
//       sequence number descending (RocksDB internal-key order).
//   (2) every iterator value self-validates via an embedded hash.
//   (3) for each user_key k, every (k, seq) committed before ReadStep
//       began is observed by the iterator — i.e. no committed write is
//       skipped by a concurrent reader.

class ConcurrentTest {
 public:
  static constexpr int K = 32;  // user_key universe size

  ConcurrentTest() : bundle_(MakeRep()) {
    for (int i = 0; i < K; i++) {
      committed_seq_[i].store(0, std::memory_order_relaxed);
    }
  }

  static std::string MakeKey(int k) {
    char buf[16];
    int n = snprintf(buf, sizeof(buf), "k_%04d", k);
    return std::string(buf, static_cast<size_t>(n));
  }

  // Value layout: <k:u32 LE><seq:u64 LE><hash:u64 LE> where the hash is
  // computed over the first 12 bytes. Lets readers detect torn writes.
  static std::string MakeValue(int k, uint64_t seq) {
    std::string v;
    PutFixed32(&v, static_cast<uint32_t>(k));
    PutFixed64(&v, seq);
    uint64_t h = Hash(v.data(), v.size(), /*seed=*/0);
    PutFixed64(&v, h);
    return v;
  }

  static bool ValidateValue(const Slice& v, int* k, uint64_t* seq) {
    if (v.size() != 4 + 8 + 8) return false;
    *k = static_cast<int>(DecodeFixed32(v.data()));
    *seq = DecodeFixed64(v.data() + 4);
    uint64_t h_stored = DecodeFixed64(v.data() + 12);
    uint64_t h_expected = Hash(v.data(), 12, /*seed=*/0);
    return h_stored == h_expected;
  }

  // REQUIRES: serialized externally (no two threads call concurrently).
  // Used by RunConcurrentRead's single writer.
  void WriteStep(Random* rnd) {
    int k = static_cast<int>(rnd->Next() % K);
    uint64_t s = global_seq_.fetch_add(1, std::memory_order_relaxed);
    InsertValue(bundle_->rep.get(), MakeKey(k), s, Slice(MakeValue(k, s)));
    committed_seq_[k].store(s, std::memory_order_release);
  }

  // Safe for any number of concurrent writers, including same-k races —
  // tests CSPP's per-VecPin LOCK_FLAG protocol.
  void ConcurrentWriteStep(int k) {
    uint64_t s = global_seq_.fetch_add(1, std::memory_order_relaxed);
    InsertValue(bundle_->rep.get(), MakeKey(k), s, Slice(MakeValue(k, s)));
    uint64_t prev = committed_seq_[k].load(std::memory_order_acquire);
    while (prev < s &&
           !committed_seq_[k].compare_exchange_weak(
               prev, s, std::memory_order_release,
               std::memory_order_acquire)) {
      // CAS failed; prev was updated, loop and retry if still < s.
    }
  }

  void ReadStep() {
    // Snapshot the latest committed seq per user_key. The release/acquire
    // pair with the writer's store-release ensures that any seq we read
    // here was inserted into the trie before our subsequent iterator was
    // constructed.
    uint64_t init[K];
    for (int i = 0; i < K; i++) {
      init[i] = committed_seq_[i].load(std::memory_order_acquire);
    }

    std::unique_ptr<MemTableRep::Iterator> it(
        bundle_->rep->GetIterator(/*arena=*/nullptr));
    uint64_t max_seen[K];
    for (int i = 0; i < K; i++) max_seen[i] = 0;

    std::string prev_user_key;
    uint64_t prev_seq = std::numeric_limits<uint64_t>::max();

    for (it->SeekToFirst(); it->Valid(); it->Next()) {
      const char* keyp = it->key();
      Slice ikey = GetLengthPrefixedSlice(keyp);
      Slice value = GetLengthPrefixedSlice(ikey.data() + ikey.size());

      ParsedInternalKey parsed;
      ASSERT_TRUE(
          ParseInternalKey(ikey, &parsed, /*log_err_key=*/false).ok());
      std::string user_key = parsed.user_key.ToString();
      uint64_t seq = parsed.sequence;

      // (1) iterator order
      if (!prev_user_key.empty()) {
        if (user_key == prev_user_key) {
          ASSERT_LT(seq, prev_seq)
              << "same-user-key entries not seq-descending";
        } else {
          ASSERT_GT(user_key, prev_user_key) << "iterator went backwards";
        }
      }
      prev_user_key = user_key;
      prev_seq = seq;

      // (2) value self-validates
      int dec_k;
      uint64_t dec_seq;
      ASSERT_TRUE(ValidateValue(value, &dec_k, &dec_seq))
          << "torn value at user_key=" << user_key << " seq=" << seq;
      ASSERT_EQ(user_key, MakeKey(dec_k));
      ASSERT_EQ(seq, dec_seq);

      if (dec_k >= 0 && dec_k < K) {
        if (seq > max_seen[dec_k]) max_seen[dec_k] = seq;
      }
    }

    // (3) nothing committed before this ReadStep was skipped
    for (int k = 0; k < K; k++) {
      if (init[k] > 0) {
        ASSERT_GE(max_seen[k], init[k])
            << "missed committed entry: k=" << k << " init=" << init[k]
            << " max_seen=" << max_seen[k];
      }
    }
  }

 private:
  std::unique_ptr<RepBundle> bundle_;
  std::atomic<uint64_t> committed_seq_[K];
  std::atomic<uint64_t> global_seq_{1};
};

constexpr int ConcurrentTest::K;

// 1 reader + 1 writer per round. The writer is on the main thread; the
// reader loops in a background thread until the writer signals quit.
void RunConcurrentRead(int run) {
  Random rnd(1000 + run * 100);
  constexpr int kRounds = 20;
  constexpr int kWritesPerRound = 500;
  for (int i = 0; i < kRounds; i++) {
    ConcurrentTest t;
    std::atomic<bool> quit{false};
    std::thread reader([&]() {
      while (!quit.load(std::memory_order_acquire)) {
        t.ReadStep();
      }
    });
    for (int j = 0; j < kWritesPerRound; j++) {
      t.WriteStep(&rnd);
    }
    quit.store(true, std::memory_order_release);
    reader.join();
  }
}

// `write_parallelism` concurrent writers + 1 reader per round. Writers
// race on the same K user_keys (same-key races allowed).
void RunConcurrentInsert(int run, int write_parallelism) {
  constexpr int kRounds = 10;
  constexpr int kWritesPerThread = 200;
  for (int i = 0; i < kRounds; i++) {
    ConcurrentTest t;
    std::atomic<bool> quit{false};
    std::thread reader([&]() {
      while (!quit.load(std::memory_order_acquire)) {
        t.ReadStep();
      }
    });
    std::vector<std::thread> writers;
    writers.reserve(static_cast<size_t>(write_parallelism));
    for (int w = 0; w < write_parallelism; w++) {
      writers.emplace_back([&t, run, i, w]() {
        Random rnd(2000 + run * 1000 + i * 100 + w);
        for (int j = 0; j < kWritesPerThread; j++) {
          int k = static_cast<int>(rnd.Next() % ConcurrentTest::K);
          t.ConcurrentWriteStep(k);
        }
      });
    }
    for (auto& w : writers) w.join();
    quit.store(true, std::memory_order_release);
    reader.join();
  }
}

TEST_F(CSPPMemTabTest, ConcurrentRead1) { RunConcurrentRead(1); }
TEST_F(CSPPMemTabTest, ConcurrentRead3) { RunConcurrentRead(3); }
TEST_F(CSPPMemTabTest, ConcurrentInsert1) { RunConcurrentInsert(1, 1); }
TEST_F(CSPPMemTabTest, ConcurrentInsert3) { RunConcurrentInsert(1, 3); }

}  // namespace ROCKSDB_NAMESPACE

int main(int argc, char** argv) {
  ROCKSDB_NAMESPACE::port::InstallStackTraceHandler();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
