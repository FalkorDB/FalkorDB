#include "RG.h"
#include "rocksdb.h"
#include <unistd.h> 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "redismodule.h"
#include "rmalloc.h"

#define ROCKSDB_PATH_BASE "/tmp/rocksdb_falkordb"

rocksdb_t *db;
rocksdb_writeoptions_t *writeoptions;
rocksdb_readoptions_t *readoptions;
rocksdb_flushoptions_t *flush_options;

void RocksDB_set_key
(
	char *node_key,
	uint64_t node_id,
	unsigned short attr_id
) {
	*(uint64_t *)node_key = node_id;
	*(unsigned short *)(node_key + 8) = attr_id;
	node_key[10] = '\0';
}

void RocksDB_init() {
	char *path = NULL;
	asprintf(&path, "%s_%d", ROCKSDB_PATH_BASE, getpid());
	char cmd[100];
	sprintf(cmd, "rm -rf %s", path);
	system(cmd);
	rocksdb_options_t *options = rocksdb_options_create();
	rocksdb_options_set_optimize_filters_for_hits(options, 1);
	rocksdb_block_based_table_options_t *table_options = rocksdb_block_based_options_create();
	rocksdb_block_based_options_set_no_block_cache(table_options, 1);
	rocksdb_options_set_block_based_table_factory(options, table_options);
	rocksdb_options_set_create_if_missing(options, 1);
	rocksdb_options_set_max_open_files(options, 100);
	rocksdb_options_set_write_buffer_size(options, 1 * 1024 * 1024);
	rocksdb_options_set_db_write_buffer_size(options, 1 * 1024 * 1024);
	rocksdb_options_set_max_bytes_for_level_base(options, 1 * 1024 * 1024);

	writeoptions = rocksdb_writeoptions_create();
	rocksdb_writeoptions_disable_WAL(writeoptions, 1);

	readoptions = rocksdb_readoptions_create();

	flush_options = rocksdb_flushoptions_create();
	rocksdb_flushoptions_set_wait(flush_options, 1);

	// open DB
	char *err = NULL;
	db = rocksdb_open(options, path, &err);
	ASSERT(!err);

	rocksdb_options_destroy(options);
	free(path);
}

rocksdb_writebatch_t *RocksDB_create_batch() {
	rocksdb_writebatch_t *writebatch = rocksdb_writebatch_create();
	return writebatch;
}

void RocksDB_put
(
	rocksdb_writebatch_t *writebatch,
	const char *key,
	const char *value
) {
	char *err = NULL;
	if(writebatch) {
		rocksdb_writebatch_put(writebatch, key, ROCKSDB_KEY_SIZE, value, strlen(value) + 1);
	} else {
		rocksdb_put(db, writeoptions, key, ROCKSDB_KEY_SIZE, value, strlen(value) + 1, &err);
	}
	ASSERT(!err);
}

void RocksDB_put_batch
(
	rocksdb_writebatch_t *writebatch
) {
	char *err = NULL;
	rocksdb_write(db, writeoptions, writebatch, &err);
	ASSERT(!err);
	rocksdb_writebatch_destroy(writebatch);
	rocksdb_flush(db, flush_options, &err);
	ASSERT(!err);
}

char *RocksDB_get
(
	const char *key
) {
	char *err = NULL;
	size_t len;
	char *returned_value = rocksdb_get(db, readoptions, key, ROCKSDB_KEY_SIZE, &len, &err);
	ASSERT(!err);
	return returned_value;
}

void print_info
(
	char *property
) {
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
	rocksdb_writeoptions_destroy(writeoptions);
	rocksdb_readoptions_destroy(readoptions);
	rocksdb_flushoptions_destroy(flush_options);
	rocksdb_close(db);
}