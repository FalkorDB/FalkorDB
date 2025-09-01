/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#include "RG.h"
#include "repository.h"
#include "../util/arr.h"
#include "../util/rmalloc.h"

#include <pthread.h>
#include <openssl/sha.h>

typedef struct {
	pthread_rwlock_t rwlock;  // read/write lock
	UDF_RepoVersion	v;        // repository version
	char **scripts;           // array of registered scripts

} UDF_Repository;

static UDF_Repository *udf_repo = NULL ;

// initialize UDF repository
bool UDF_RepoInit(void) {
	ASSERT (udf_repo == NULL) ;

	udf_repo = rm_calloc (1, sizeof(UDF_Repository)) ;

	// repo read/write lock
	int res = pthread_rwlock_init (&udf_repo->rwlock, NULL) ;
	if (res) {
		return false ;
	}

	udf_repo->scripts = array_new(char*, 1) ;

	return (udf_repo->scripts != NULL) ;
}

// return repo's version
UDF_RepoVersion UDF_RepoGetVersion(void) {
	ASSERT (udf_repo != NULL) ;

	return udf_repo->v ;
}

// build a new JSContext for the given JSRuntime
// the new js context will be loaded with all registered scripts
JSContext *UDF_RepoBuildJSContext
(
	JSRuntime *js_rt,   // javascript runtime
	UDF_RepoVersion *v  // [output] repo version
) {
	ASSERT (v        != NULL) ;
	ASSERT (js_rt    != NULL) ;
	ASSERT (udf_repo != NULL) ;

	// create js context
	JSContext *js_ctx = JS_NewContext(js_rt) ;

	// lock under READ
	pthread_rwlock_rdlock (&udf_repo->rwlock) ;

	// load each registered script
	int n = array_len (udf_repo->scripts) ;
	for (int i = 0; i < n; i++) {
		const char *script = udf_repo->scripts[i] ;

		// evalute script
		JSValue val = JS_Eval (js_ctx, script, strlen(script), "<input>",
				JS_EVAL_TYPE_GLOBAL) ;

		ASSERT (!JS_IsException (val)) ;

	}

	// set version
	*v = udf_repo->v ;

	// unlock
	pthread_rwlock_unlock (&udf_repo->rwlock) ;

	return js_ctx ;
}

// returns script from UDF repository
const char *UDF_RepoGetScript
(
	const unsigned char *hash,  // script SHA1 hash to retrieve
	int *idx                    // [optional] script index
) {
	ASSERT (hash     != NULL) ;
	ASSERT (udf_repo != NULL) ;

	const char *script = NULL ;

	pthread_rwlock_rdlock (&udf_repo->rwlock) ;

	int n = array_len(udf_repo->scripts) ;
	for (int i = 0; i < n; i++) {
		const char *_script = udf_repo->scripts[i] ;

		// compute script hash
		unsigned char digest [SHA_DIGEST_LENGTH] ;
		SHA1 ((unsigned char*)_script, strlen (_script), digest) ;

		if (memcmp (hash, digest, SHA_DIGEST_LENGTH) == 0) {
			script = _script ;

			if (idx != NULL) {
				*idx = i ;
			}

			break ;
		}
	}

	pthread_rwlock_unlock (&udf_repo->rwlock) ;

	return script ;
}

// checks if UDF repository contains script
bool UDF_RepoContainsScript
(
	const unsigned char *hash,  // script SHA1 hash to look for
	int *idx                    // [optional] script index
) {
	ASSERT (hash     != NULL) ;
	ASSERT (udf_repo != NULL) ;

	return (UDF_RepoGetScript (hash, idx) != NULL) ;
}

// register a new UDF script
bool UDF_RepoRegisterScript
(
	const char *script  // script
) {
	ASSERT (script   != NULL) ;
	ASSERT (udf_repo != NULL) ;

	// compute script's sha-1 hash
	unsigned char hash[SHA_DIGEST_LENGTH];  // SHA1 outputs 20 bytes
	SHA1((unsigned char*)script, strlen(script), hash);

	if (UDF_RepoContainsScript (hash, NULL)) {
		return false ;
	}

	// lock under write
	pthread_rwlock_wrlock (&udf_repo->rwlock) ;

	array_append (udf_repo->scripts, rm_strdup (script)) ;

	// bump version
	udf_repo->v++ ;

	// unlock
	pthread_rwlock_unlock (&udf_repo->rwlock) ;

	return true ;
}

// removes a script from UDF repository
bool UDF_RepoRemoveScript
(
	const unsigned char *hash,  // script SHA1 hash to remove
	char **script               // removed script
) {
	ASSERT (hash     != NULL) ;
	ASSERT (udf_repo != NULL) ;
	
	int   idx     = -1;
	bool  res     = false ;
	char *_script = NULL ;

	if (UDF_RepoContainsScript (hash, &idx)) {
		// lock under WRITE
		pthread_rwlock_wrlock (&udf_repo->rwlock) ;

		_script = udf_repo->scripts[idx] ;

		array_del (udf_repo->scripts, idx);

		udf_repo->v++ ; // bump version
		res = true ;

		// unlock
		pthread_rwlock_unlock (&udf_repo->rwlock) ;
	}

	// in case script was found, free or return it
	if (res) {
		if (script != NULL) {
			*script = _script ;
		} else {
			rm_free (_script) ;
		}
	}

	return res ;
}

// free UDF repository
void UDF_RepoFree(void) {
	if (udf_repo == NULL) {
		return ;
	}

	for (int i = 0; i < array_len(udf_repo->scripts); i++) {
		rm_free (udf_repo->scripts[i]) ;
	}

	array_free (udf_repo->scripts) ;

	// free lock
	int res = pthread_rwlock_destroy (&udf_repo->rwlock) ;
	ASSERT (res == 0) ;

	rm_free (udf_repo) ;
}

