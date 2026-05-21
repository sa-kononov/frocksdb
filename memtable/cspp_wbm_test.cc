// SPDX-License-Identifier: GPL-2.0 OR Apache-2.0
// Verifies that CSPPMemTab forwards mempool bytes to WriteBufferManager
// and (when the WBM was constructed with a Cache) that those bytes appear
// in the shared cache as reserved entries — the "cost-to-cache" model
// Flink relies on.
//
// Build:   make WITH_CSPP_MEMTABLE=1 cspp_wbm_test
// Run:     ./cspp_wbm_test

#include <cstring>
#include <memory>
#include <string>

#include "db/dbformat.h"
#include "memory/arena.h"
#include "port/stack_trace.h"
#include "rocksdb/advanced_cache.h"
#include "rocksdb/cache.h"
#include "rocksdb/comparator.h"
#include "rocksdb/memtablerep.h"
#include "rocksdb/slice.h"
#include "rocksdb/slice_transform.h"
#include "rocksdb/write_buffer_manager.h"
#include "test_util/testharness.h"
#include "util/coding.h"

namespace ROCKSDB_NAMESPACE {

namespace {

constexpr size_t kMiB = 1ull << 20;

// MemTableRep::KeyComparator backed by an InternalKeyComparator. CSPP itself
// ignores the comparator (it always sorts byte-wise), but CreateMemTableRep
// requires one of these.
struct InternalKeyComparatorAdapter : public MemTableRep::KeyComparator {
  const InternalKeyComparator* cmp;
  explicit InternalKeyComparatorAdapter(const InternalKeyComparator* c)
      : cmp(c) {}
  int operator()(const char* a, const char* b) const override {
    Slice ka = GetLengthPrefixedSlice(a);
    Slice kb = GetLengthPrefixedSlice(b);
    return cmp->Compare(ka, kb);
  }
  int operator()(const char* a, const Slice& kb) const override {
    Slice ka = GetLengthPrefixedSlice(a);
    return cmp->Compare(ka, kb);
  }
};

struct RepBundle {
  std::unique_ptr<MemTableRepFactory> factory;
  std::unique_ptr<Arena> arena;
  std::unique_ptr<InternalKeyComparator> ikc;
  std::unique_ptr<InternalKeyComparatorAdapter> cmp;
  std::unique_ptr<MemTableRep> rep;
  uint64_t next_seq = 1;
};

std::unique_ptr<RepBundle> MakeRep(size_t mem_cap,
                                   WriteBufferManager* wbm = nullptr,
                                   bool read_by_writer_token = true) {
  auto bundle = std::make_unique<RepBundle>();
  bundle->factory.reset(
      NewCSPPMemTableRepFactory(mem_cap, read_by_writer_token, wbm));
  EXPECT_NE(bundle->factory, nullptr)
      << "NewCSPPMemTableRepFactory returned nullptr (was the build "
         "configured with WITH_CSPP_MEMTABLE=1?)";
  bundle->arena = std::make_unique<Arena>();
  bundle->ikc = std::make_unique<InternalKeyComparator>(BytewiseComparator());
  bundle->cmp =
      std::make_unique<InternalKeyComparatorAdapter>(bundle->ikc.get());
  bundle->rep.reset(bundle->factory->CreateMemTableRep(
      *bundle->cmp, bundle->arena.get(), /*transform=*/nullptr,
      /*logger=*/nullptr));
  EXPECT_NE(bundle->rep, nullptr) << "CreateMemTableRep returned nullptr";
  return bundle;
}

void InsertOne(MemTableRep* rep, uint64_t* seq, const std::string& user_key,
               const Slice& value) {
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
  // Tag = (seq << 8) | kTypeValue
  EncodeFixed64(p, ((*seq)++ << 8) | 0x1);
  p += 8;
  p = EncodeVarint32(p, static_cast<uint32_t>(value.size()));
  memcpy(p, value.data(), value.size());
  rep->Insert(handle);
}

// Writes `target_bytes` of unique 1 KiB values into `rep`, invoking
// `progress_cb(bytes_written)` after every `progress_step` bytes. `key_prefix`
// namespaces the keys so multiple reps in one process get disjoint keyspaces.
template <typename ProgressCb>
size_t WriteApprox(RepBundle* bundle, const std::string& key_prefix,
                   size_t target_bytes, size_t progress_step,
                   ProgressCb&& progress_cb) {
  const size_t kValSize = 1024;
  std::string value(kValSize, 'x');
  for (size_t i = 0; i < kValSize; i++) {
    value[i] = static_cast<char>('a' + (i % 26));
  }
  size_t bytes_written = 0;
  size_t next_progress = progress_step;
  size_t i = 0;
  char keybuf[64];
  while (bytes_written < target_bytes) {
    int n = snprintf(keybuf, sizeof(keybuf), "%s_%012zu", key_prefix.c_str(),
                     i++);
    EXPECT_GT(n, 0);
    EXPECT_LT(static_cast<size_t>(n), sizeof(keybuf));
    std::string ukey(keybuf, static_cast<size_t>(n));
    InsertOne(bundle->rep.get(), &bundle->next_seq, ukey, Slice(value));
    bytes_written += ukey.size() + kValSize;
    if (bytes_written >= next_progress) {
      progress_cb(bytes_written);
      next_progress += progress_step;
    }
  }
  return bytes_written;
}

std::shared_ptr<Cache> MakeCache(size_t capacity) {
  LRUCacheOptions co(capacity, /*num_shard_bits=*/-1,
                     /*strict_capacity_limit=*/false,
                     /*high_pri_pool_ratio=*/0.5);
  return NewLRUCache(co);
}

}  // namespace

class CSPPMemTabWbmTest : public testing::Test {};

TEST_F(CSPPMemTabWbmTest, WbmTracksTrieGrowth) {
  auto cache = MakeCache(256 * kMiB);
  ASSERT_NE(cache, nullptr);

  WriteBufferManager wbm(/*buffer_size=*/64 * kMiB, cache);
  EXPECT_TRUE(wbm.enabled());
  EXPECT_TRUE(wbm.cost_to_cache());

  const size_t kMemCap = 128 * kMiB;
  auto bundle = MakeRep(kMemCap, &wbm);
  ASSERT_NE(bundle, nullptr);

  size_t last_usage = 0;
  bool monotonic = true;
  WriteApprox(bundle.get(), "t1", /*target=*/10 * kMiB, /*step=*/1 * kMiB,
              [&](size_t /*bytes*/) {
                size_t cur = wbm.memory_usage();
                if (cur < last_usage) {
                  monotonic = false;
                }
                last_usage = cur;
              });
  EXPECT_TRUE(monotonic)
      << "wbm.memory_usage() must be monotonically non-decreasing";

  const size_t final_usage = wbm.memory_usage();
  // Patricia might compress; require at least 8 MiB for a 10 MiB raw insert.
  EXPECT_GE(final_usage, 8 * kMiB);
  EXPECT_LE(final_usage, kMemCap);
}

TEST_F(CSPPMemTabWbmTest, CacheReservationTracksWbm) {
  auto cache = MakeCache(256 * kMiB);
  WriteBufferManager wbm(/*buffer_size=*/64 * kMiB, cache);

  auto bundle = MakeRep(/*mem_cap=*/128 * kMiB, &wbm);

  WriteApprox(bundle.get(), "t2", /*target=*/10 * kMiB, /*step=*/1 * kMiB,
              [](size_t) {});

  const size_t wbm_bytes = wbm.memory_usage();
  const size_t cache_bytes = cache->GetUsage();

  // WBM reserves dummy entries in 1 MiB chunks; a ~10 MiB burst should add
  // several MiB of reservations to the cache.
  EXPECT_GE(cache_bytes, 1 * kMiB);
  // WBM should not massively over-reserve. Allow up to 2 MiB slack for the
  // 1 MiB chunk rounding plus any cache metadata overhead.
  EXPECT_LE(cache_bytes, wbm_bytes + 2 * kMiB);
}

TEST_F(CSPPMemTabWbmTest, CrossMemtableAggregation) {
  auto cache = MakeCache(256 * kMiB);
  WriteBufferManager wbm(/*buffer_size=*/48 * kMiB, cache);

  auto bundle_a = MakeRep(/*mem_cap=*/64 * kMiB, &wbm);
  auto bundle_b = MakeRep(/*mem_cap=*/64 * kMiB, &wbm);

  WriteApprox(bundle_a.get(), "t3a", /*target=*/5 * kMiB, /*step=*/1 * kMiB,
              [](size_t) {});
  WriteApprox(bundle_b.get(), "t3b", /*target=*/5 * kMiB, /*step=*/1 * kMiB,
              [](size_t) {});

  const size_t combined = wbm.memory_usage();
  EXPECT_GE(combined, 8 * kMiB);
  EXPECT_LE(combined, 20 * kMiB);

  const size_t a_size = bundle_a->rep->ApproximateMemoryUsage();
  bundle_a.reset();
  const size_t after = wbm.memory_usage();
  // After releasing rep A, wbm should have dropped by roughly a_size. Allow
  // wide slack — patricia's mem_size_inline is an approximation and WBM
  // reservation rounds to 1 MiB.
  EXPECT_GE(after + a_size, combined - 2 * kMiB);
  EXPECT_LE(after, combined);
}

TEST_F(CSPPMemTabWbmTest, FreeMemOnDestruction) {
  auto cache = MakeCache(256 * kMiB);
  WriteBufferManager wbm(/*buffer_size=*/64 * kMiB, cache);

  auto bundle = MakeRep(/*mem_cap=*/64 * kMiB, &wbm);
  WriteApprox(bundle.get(), "t4", /*target=*/5 * kMiB, /*step=*/1 * kMiB,
              [](size_t) {});

  const size_t pre = wbm.memory_usage();
  ASSERT_GE(pre, 4 * kMiB);

  bundle.reset();

  EXPECT_EQ(wbm.memory_usage(), 0u);
  // Cache may retain a sub-chunk briefly depending on WBM bookkeeping;
  // require it to be < 1 MiB after release.
  EXPECT_LT(cache->GetUsage(), 1 * kMiB);
}

TEST_F(CSPPMemTabWbmTest, NoWbmCompatibility) {
  // No WBM attached.
  auto bundle = MakeRep(/*mem_cap=*/64 * kMiB);
  WriteApprox(bundle.get(), "t5", /*target=*/5 * kMiB, /*step=*/1 * kMiB,
              [](size_t) {});
  EXPECT_GE(bundle->rep->ApproximateMemoryUsage(), 4 * kMiB);
  // Destroying with no WBM attached must not crash.
  bundle.reset();
}

}  // namespace ROCKSDB_NAMESPACE

int main(int argc, char** argv) {
  ROCKSDB_NAMESPACE::port::InstallStackTraceHandler();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
