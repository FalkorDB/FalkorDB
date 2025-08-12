/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#include "RG.h"
#include "../util/dict.h"
#include "../util/rmalloc.h"
#include "deps/xxHash/xxhash.h"

#include <pthread.h>

typedef struct {
	dict *repo;
	pthread_mutex_t lock;
} UDF_Repository;

static UDF_Repository *udf_repo = NULL ;

// hashtable callbacks

// key hash function
static uint64_t _hashFunc
(
	const void *key
) {
	const char *str = (const char *)key ;
	return XXH3_64bits (key, strlen(key)) ;
}

// key compare function
static int _keyCompare
(
	dict *d,
	const void *key1,
	const void *key2
) {
	return strcmp ((const char*)key1, (const char*)key2) == 0 ;
}

static void *_keyDup
(
	dict *d,
	const void *key
) {
	ASSERT (key != NULL) ;
	return rm_strdup ((const char *)key) ;
}

// key free function
static void _keyDestructor
(
	dict *d,
	void *key
) {
	ASSERT (key != NULL) ;

	rm_free (key) ;
}

static void *_valDup
(
	dict *d,
	const void *obj
) {
	ASSERT (obj != NULL) ;
	return rm_strdup ((const char *)obj) ;
}

// key free function
static void _valDestructor
(
	dict *d,
	void *val
) {
	ASSERT (val != NULL) ;

	rm_free (val) ;
}

// StringPool hashtable callbacks
static const dictType _dictType = {
	.hashFunction           = _hashFunc,
	.keyDup                 = _keyDup,
	.valDup                 = _valDup,
	.keyCompare             = _keyCompare,
	.keyDestructor          = _keyDestructor,
	.valDestructor          = _valDestructor,
	.expandAllowed          = NULL,
	.dictEntryMetadataBytes = NULL,
	.dictMetadataBytes      = NULL,
	.afterReplaceEntry      = NULL
};

// initialize UDF repository
bool UDF_RepoInit(void) {
	ASSERT (udf_repo == NULL) ;

	udf_repo = rm_calloc (1, sizeof(UDF_Repository)) ;

	int res = pthread_mutex_init (&udf_repo->lock, NULL) ;
	if (res) {
		return false ;
	}

	udf_repo->repo = HashTableCreate (&_dictType) ;
	return (udf_repo->repo != NULL) ;
}

// search UDF repository for function
const char *UDF_RepoGetScript
(
	const char *func_name  // function name to look for
) {
	ASSERT (func_name != NULL) ;

	const char *script = NULL ;

	pthread_mutex_lock (&udf_repo->lock) ;

	dictEntry *e = HashTableFind (udf_repo->repo, func_name) ;
	if (e != NULL) {
		script = HashTableGetVal(e) ;
	}

	pthread_mutex_unlock (&udf_repo->lock) ;

	return script ;
}

// returns true if UDF repository contains script
bool UDF_RepoContainsScript
(
	const char *func_name  // function name to look for
) {
	return (UDF_RepoGetScript (func_name) != NULL) ;
}

bool UDF_RepoSetScript
(
	const char *func_name,
	const char *script
) {
	ASSERT (script    != NULL) ;
	ASSERT (func_name != NULL) ;

	pthread_mutex_lock (&udf_repo->lock) ;

	dictEntry *de = HashTableAddRaw (udf_repo->repo, (void*)func_name, NULL) ;
	if (de == NULL) {
		// function already registered
		return false ;
	}

	HashTableSetVal(udf_repo->repo, de, (void*) script) ;

	pthread_mutex_unlock (&udf_repo->lock) ;

	return true ;
}

bool UDF_UnRegisterScript
(
	const char *func_name
) {
	ASSERT (func_name != NULL) ;
	
	pthread_mutex_lock (&udf_repo->lock) ;

	int res = HashTableDelete (udf_repo->repo, func_name) ;

	pthread_mutex_unlock (&udf_repo->lock) ;

	return res == DICT_OK ;
}

void UDF_RepoFree(void) {
	if (udf_repo == NULL) {
		return ;
	}

	int res = pthread_mutex_destroy (&udf_repo->lock) ;
	ASSERT (res == 0) ;
	
	HashTableRelease (udf_repo->repo) ;

	rm_free (udf_repo) ;
}

