#include "RG.h"
#include "rocksdb.h"
#include <unistd.h> 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <rocksdb/c.h>
#include "redismodule.h"

const char DBPath[] = "/tmp/rocksdb_falkordb";

rocksdb_t *db;

void RocksDB_init() {
	rocksdb_options_t *options = rocksdb_options_create();
	rocksdb_options_set_optimize_filters_for_hits(options, 1);
	rocksdb_block_based_table_options_t *table_options = rocksdb_block_based_options_create();
	rocksdb_block_based_options_set_no_block_cache(table_options, 1);
	rocksdb_options_set_block_based_table_factory(options, table_options);
	rocksdb_options_set_create_if_missing(options, 1);
	rocksdb_options_set_max_open_files(options, 100);
	rocksdb_options_set_write_buffer_size(options, 1 * 1024 * 1024);
	rocksdb_options_set_db_write_buffer_size(options, 1 * 1024 * 1024);

	// open DB
	char *err = NULL;
	db = rocksdb_open(options, DBPath, &err);
	ASSERT(!err);
}

rocksdb_writebatch_t *RocksDB_create_batch() {
	rocksdb_writebatch_t *writebatch = rocksdb_writebatch_create();
	return writebatch;
}

void RocksDB_put(rocksdb_writebatch_t *writebatch, const char *key, const char *value) {
	rocksdb_writeoptions_t *writeoptions = rocksdb_writeoptions_create();
	char *err = NULL;
	if(writebatch) {
		rocksdb_writebatch_put(writebatch, key, strlen(key), value, strlen(value) + 1);
	} else {
		rocksdb_put(db, writeoptions, key, strlen(key), value, strlen(value) + 1, &err);
	}
	ASSERT(!err);
	rocksdb_writeoptions_destroy(writeoptions);
}

void RocksDB_put_batch(rocksdb_writebatch_t *writebatch) {
	rocksdb_writeoptions_t *writeoptions = rocksdb_writeoptions_create();
	char *err = NULL;
	rocksdb_write(db, writeoptions, writebatch, &err);
	ASSERT(!err);
	rocksdb_writeoptions_destroy(writeoptions);
	rocksdb_writebatch_destroy(writebatch);

}

char *RocksDB_get(const char *key) {
	rocksdb_readoptions_t *readoptions = rocksdb_readoptions_create();
	char *err = NULL;
	size_t len;
	char *returned_value = rocksdb_get(db, readoptions, key, strlen(key), &len, &err);
	ASSERT(!err);
	rocksdb_readoptions_destroy(readoptions);
	return returned_value;
}

void print_info(char *property) {
	char* value = rocksdb_property_value(db, property);
    if (value != NULL) {
        RedisModule_Log(NULL, "notice", "%s: %s\n", property, value);
        free(value); // Free the returned value
    } else {
        RedisModule_Log(NULL, "notice", "Failed to retrieve %s", property);
    }
}

void RocksDB_info() {
	print_info("rocksdb.block-cache-usage");
	print_info("rocksdb.cur-size-all-mem-tables");
	print_info("rocksdb.estimate-table-readers-mem");
	print_info("rocksdb.block-cache-pinned-usage");
}

void RocksDB_cleanup() {
	rocksdb_close(db);
}