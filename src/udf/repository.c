/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#include "RG.h"
#include "../util/dict.h"
#include "../util/rmalloc.h"

#include <pthread.h>
#include <openssl/sha.h>

typedef struct {
	dict *repo;
	pthread_mutex_t lock;
} UDF_Repository;

static UDF_Repository *udf_repo = NULL ;

// hashtable callbacks

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
	.keyDup                 = _keyDup,
	.valDup                 = _valDup,
	.keyCompare             = _keyCompare,
	.keyDestructor          = _keyDestructor,
	.valDestructor          = _valDestructor,
	.hashFunction           = NULL,
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

// returns script from UDF repository
const char *UDF_RepoGetScript
(
	const unsigned char *hash  // script SHA1 hash to retrieve
) {
	ASSERT (hash != NULL) ;

	const char *script = NULL ;

	pthread_mutex_lock (&udf_repo->lock) ;

	dictEntry *e = HashTableFind (udf_repo->repo, hash) ;
	if (e != NULL) {
		script = HashTableGetVal(e) ;
	}

	pthread_mutex_unlock (&udf_repo->lock) ;

	return script ;
}

// checks if UDF repository contains script
bool UDF_RepoContainsScript
(
	const unsigned char *hash  // script SHA1 hash to look for
) {
	return (UDF_RepoGetScript (hash) != NULL) ;
}

// register a new UDF script
bool UDF_RepoRegisterScript
(
	const char *script  // script
) {
	ASSERT (script != NULL) ;

	// compute script's sha-1 hash
	unsigned char hash[SHA_DIGEST_LENGTH];  // SHA1 outputs 20 bytes
    SHA1((unsigned char*)script, strlen(script), hash);

	pthread_mutex_lock (&udf_repo->lock) ;

	dictEntry *de = HashTableAddRaw (udf_repo->repo, (void*)hash, NULL) ;
	if (de == NULL) {
		// function already registered
		return false ;
	}

	HashTableSetVal(udf_repo->repo, de, (void*) script) ;

	pthread_mutex_unlock (&udf_repo->lock) ;

	return true ;
}

// removes a script from UDF repository
bool UDF_RepoRemoveScript
(
	const char *hash  // script SHA1 hash to remove
) {
	ASSERT (hash != NULL) ;
	
	pthread_mutex_lock (&udf_repo->lock) ;

	int res = HashTableDelete (udf_repo->repo, hash) ;

	pthread_mutex_unlock (&udf_repo->lock) ;

	return res == DICT_OK ;
}

// free UDF repository
void UDF_RepoFree(void) {
	if (udf_repo == NULL) {
		return ;
	}

	int res = pthread_mutex_destroy (&udf_repo->lock) ;
	ASSERT (res == 0) ;
	
	HashTableRelease (udf_repo->repo) ;

	rm_free (udf_repo) ;
}

