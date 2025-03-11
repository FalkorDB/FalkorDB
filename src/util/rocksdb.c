#include "RG.h"
#include "rocksdb.h"
#include <unistd.h> 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <rocksdb/c.h>
#include "redismodule.h"

const char DBPath[] = "/tmp/rocksdb_falkordb";
const char DBBackupPath[] = "/tmp/rocksdb_falkordb_backup";

rocksdb_t *db;
rocksdb_backup_engine_t *be;

void RocksDB_init() {
	rocksdb_options_t *options = rocksdb_options_create();
	// Optimize RocksDB. This is the easiest way to
	// get RocksDB to perform well.
	// long cpus = sysconf(_SC_NPROCESSORS_ONLN);
	// // Set # of online cores
	// rocksdb_options_increase_parallelism(options, (int)(cpus));
	rocksdb_options_increase_parallelism(options, 1);
	rocksdb_options_optimize_level_style_compaction(options, 0);
	rocksdb_block_based_table_options_t *table_options = rocksdb_block_based_options_create();
	rocksdb_block_based_options_set_block_size(table_options, 4096);
	rocksdb_options_set_block_based_table_factory(options, table_options);
	// create the DB if it's not already present
	rocksdb_options_set_create_if_missing(options, 1);
	rocksdb_options_set_max_open_files(options, 100);
	rocksdb_options_set_write_buffer_size(options, 16 * 1024 * 1024);
	rocksdb_options_set_max_write_buffer_number(options, 3);

	// open DB
	char *err = NULL;
	db = rocksdb_open(options, DBPath, &err);
	ASSERT(!err);
  
	// // open Backup Engine that we will use for backing up our database
	// be = rocksdb_backup_engine_open(options, DBBackupPath, &err);
	// ASSERT(!err);
  
	// // create new backup in a directory specified by DBBackupPath
	// rocksdb_backup_engine_create_new_backup(be, db, &err);
	// ASSERT(!err);
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

void RocksDB_info() {
	const char* property_name = "rocksdb.block-cache-usage";
    char* value = rocksdb_property_value(db, property_name);
    if (value != NULL) {
        RedisModule_Log(NULL, "notice", "Block cache usage: %s bytes\n", value);
        free(value); // Free the returned value
    } else {
        RedisModule_Log(NULL, "notice", "Failed to retrieve block cache usage.\n");
    }
}

void RocksDB_cleanup() {
	rocksdb_backup_engine_close(be);
	rocksdb_close(db);
}