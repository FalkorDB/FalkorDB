#pragma once

#include <rocksdb/c.h>

#define ROCKSDB_MIN_STR_LEN 33

void RocksDB_init();
rocksdb_writebatch_t *RocksDB_create_batch();
void RocksDB_put(rocksdb_writebatch_t *writebatch, const char *key, const char *value);
void RocksDB_put_batch(rocksdb_writebatch_t *writebatch);
char *RocksDB_get(const char *key);
void RocksDB_info();
void RocksDB_cleanup();