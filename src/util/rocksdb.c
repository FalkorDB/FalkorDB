#include "RG.h"
#include "rocksdb.h"
#include <unistd.h> 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>
#include "redismodule.h"
#include "rmalloc.h"
#include "../configuration/config.h"

#define ROCKSDB_PATH_BASE "/tmp/rocksdb_falkordb"

bool use_disk_storage;
uint64_t value_spill_threshold;

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
}

bool RocksDB_shouldWrite
(
    const char *s
) {
	return use_disk_storage && 
		strnlen(s, value_spill_threshold) == value_spill_threshold;
}

void RocksDB_init() {
	Config_Option_get(Config_USE_DISK_STORAGE, &use_disk_storage);
	if(!use_disk_storage) return;
	Config_Option_get(Config_VALUE_SPILL_THRESHOLD, &value_spill_threshold);
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
	rocksdb_options_set_compression(options, rocksdb_no_compression);

	writeoptions = rocksdb_writeoptions_create();
	rocksdb_writeoptions_disable_WAL(writeoptions, 1);

	readoptions = rocksdb_readoptions_create();

	flush_options = rocksdb_flushoptions_create();
	rocksdb_flushoptions_set_wait(flush_options, 1);

	// open DB
	char *err = NULL;
	db = rocksdb_open(options, path, &err);
	ASSERT(!err);

	rocksdb_block_based_options_destroy(table_options);
	rocksdb_options_destroy(options);
	free(path);
}

rocksdb_writebatch_t *RocksDB_create_batch() {
	if(!use_disk_storage) return NULL;

	rocksdb_writebatch_t *writebatch = rocksdb_writebatch_create();
	return writebatch;
}

void RocksDB_put
(
	rocksdb_writebatch_t *writebatch,
	const char *key,
	const char *value
) {
	if(!use_disk_storage) return;

	char *err = NULL;
	if(writebatch) {
		rocksdb_writebatch_put(writebatch, key, ROCKSDB_KEY_SIZE, value, strlen(value) + 1);
	} else {
		rocksdb_put(db, writeoptions, key, ROCKSDB_KEY_SIZE, value, strlen(value) + 1, &err);
	}
	ASSERT(!err);
}

void RocksDB_del
(
    rocksdb_writebatch_t *writebatch,
    const char *key
) {
	if(!use_disk_storage) return;

	char *err = NULL;
	if(writebatch) {
		rocksdb_writebatch_delete(writebatch, key, ROCKSDB_KEY_SIZE);
	} else {
		rocksdb_delete(db, writeoptions, key, ROCKSDB_KEY_SIZE, &err);
	}
	ASSERT(!err);
}

void RocksDB_put_batch
(
	rocksdb_writebatch_t *writebatch
) {
	if(!use_disk_storage) return;

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
	ASSERT(use_disk_storage);
	if(!use_disk_storage) return NULL;

	char *err = NULL;
	size_t len;
	char *returned_value = rocksdb_get(db, readoptions, key, ROCKSDB_KEY_SIZE, &len, &err);
	ASSERT(!err);
	return returned_value;
}

uint64_t get_sst_file_size(const char *directory) {
    struct dirent *entry;
    struct stat file_stat;
    uint64_t total_size = 0;

    DIR *dir = opendir(directory);
    if (!dir) {
        perror("Failed to open directory");
        return 0;
    }

    while ((entry = readdir(dir)) != NULL) {
        if (strstr(entry->d_name, ".sst")) { // Check if file is an SST file
            char file_path[1024];
            snprintf(file_path, sizeof(file_path), "%s/%s", directory, entry->d_name);

            if (stat(file_path, &file_stat) == 0) {
                total_size += file_stat.st_size;
            }
        }
    }

    closedir(dir);
    return total_size;
}

void RocksDB_get_info(char **num_keys, uint64_t *disk_usage) {
	if(!use_disk_storage) return;

	*num_keys = rocksdb_property_value(db, "rocksdb.estimate-num-keys");
	char *path = NULL;
	asprintf(&path, "%s_%d", ROCKSDB_PATH_BASE, getpid());
	*disk_usage = get_sst_file_size(path);
	free(path);
}

void RocksDB_cleanup() {
	if(!use_disk_storage) return;

	rocksdb_writeoptions_destroy(writeoptions);
	rocksdb_readoptions_destroy(readoptions);
	rocksdb_flushoptions_destroy(flush_options);
	rocksdb_close(db);
}