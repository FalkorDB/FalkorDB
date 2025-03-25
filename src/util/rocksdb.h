#pragma once

#include <rocksdb/c.h>

#define ROCKSDB_MIN_STR_LEN 33
#define ROCKSDB_KEY_SIZE 11

void RocksDB_set_key
(
    char *node_key,
    uint64_t node_id,
    unsigned short attr_id
);

void RocksDB_init();

rocksdb_writebatch_t *RocksDB_create_batch();

void RocksDB_put
(
    rocksdb_writebatch_t *writebatch,
    const char *key,
    const char *value
);

void RocksDB_put_batch
(
    rocksdb_writebatch_t *writebatch
);

char *RocksDB_get(
    const char *key
);

void RocksDB_info();

void RocksDB_cleanup();
