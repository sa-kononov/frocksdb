// Copyright (c) 2026 — CSPP memtable port from ToplingDB.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
package org.rocksdb;

/**
 * The config for the CSPP (Crash-Safe Persistent Patricia trie) memtable.
 *
 * <p>The CSPP memtable stores keys in a Patricia trie, giving compact memory
 * representation for keys with shared prefixes — a common pattern for stream
 * processing workloads such as Apache Flink statebackend. The trie supports
 * lock-free concurrent inserts.
 *
 * <p>This class is only usable when the underlying native library was built
 * with {@code WITH_CSPP_MEMTABLE=1}. Otherwise, calls to
 * {@link #newMemTableFactoryHandle()} will fail with
 * {@link UnsatisfiedLinkError}.
 *
 * @see Options#setMemTableConfig(MemTableConfig)
 */
public class CSPPMemTableConfig extends MemTableConfig {

  /** Default initial trie capacity (bytes). 0 selects an internal default. */
  public static final long DEFAULT_MEM_CAP = 0;

  /** Default value of read_by_writer_token. */
  public static final boolean DEFAULT_READ_BY_WRITER_TOKEN = true;

  public CSPPMemTableConfig() {
    memCap_ = DEFAULT_MEM_CAP;
    readByWriterToken_ = DEFAULT_READ_BY_WRITER_TOKEN;
  }

  /**
   * Sets the initial trie capacity in bytes. Set to 0 to use the internal
   * default (currently 2 GiB).
   *
   * @param memCap initial trie capacity in bytes
   * @return the current instance of CSPPMemTableConfig
   */
  public CSPPMemTableConfig setMemCap(final long memCap) {
    memCap_ = memCap;
    return this;
  }

  /** @return current initial trie capacity (bytes) */
  public long memCap() {
    return memCap_;
  }

  /**
   * Controls the token type used for point lookups. When {@code true},
   * lookups reuse the thread-local writer token (cheaper on write-heavy
   * workloads); when {@code false}, a separate reader token is used (cheaper
   * on read-heavy workloads).
   *
   * @param readByWriterToken whether to read via the writer token
   * @return the current instance of CSPPMemTableConfig
   */
  public CSPPMemTableConfig setReadByWriterToken(
      final boolean readByWriterToken) {
    readByWriterToken_ = readByWriterToken;
    return this;
  }

  /** @return current read_by_writer_token setting */
  public boolean readByWriterToken() {
    return readByWriterToken_;
  }

  /**
   * Attach a {@link WriteBufferManager} so that CSPP memtable bytes are
   * forwarded to it via ReserveMem/FreeMem. This makes the memtable
   * participate in a process-wide memory budget (e.g., Flink's managed
   * memory) and, when the WriteBufferManager was constructed with a Cache,
   * also charges those bytes against that cache (cost-to-cache).
   *
   * <p>The caller is responsible for keeping {@code wbm} alive for as long
   * as any RocksDB instance using this memtable config is open. Typically
   * the WriteBufferManager is owned by the same higher-level component that
   * owns the RocksDB instances (e.g., Flink's RocksDBSharedResources).
   *
   * @param wbm the WriteBufferManager to forward byte accounting to, or
   *            {@code null} to disable forwarding
   * @return the current instance of CSPPMemTableConfig
   */
  public CSPPMemTableConfig setWriteBufferManager(final WriteBufferManager wbm) {
    this.writeBufferManagerHandle_ = (wbm == null) ? 0L : wbm.nativeHandle_;
    return this;
  }

  @Override
  protected long newMemTableFactoryHandle() {
    return newMemTableFactoryHandle0(memCap_, readByWriterToken_, writeBufferManagerHandle_);
  }

  private native long newMemTableFactoryHandle0(long memCap,
                                                boolean readByWriterToken,
                                                long writeBufferManagerHandle)
      throws IllegalArgumentException;

  private long memCap_;
  private boolean readByWriterToken_;
  private long writeBufferManagerHandle_ = 0;
}
