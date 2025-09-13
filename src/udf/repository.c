/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#include "RG.h"
#include "classes.h"
#include "repository.h"
#include "../util/arr.h"
#include "../util/rmalloc.h"
#include "../arithmetic/func_desc.h"

#include <pthread.h>
#include <openssl/sha.h>

typedef struct {
	char *name;        // library name
	char **functions;  // library functions
	char *script;      // library script
} UDF_Lib;

typedef struct {
	pthread_rwlock_t rwlock;  // read/write lock
	UDF_RepoVersion	v;        // repository version
	UDF_Lib *libs;            // array of registered UDF libs
} UDF_Repository;

static UDF_Repository *udf_repo = NULL ;

// free a registered library
void static UDF_Lib_Free
(
	UDF_Lib *lib  // library to free
) {
	ASSERT (lib != NULL) ;

	// free each registered function
	if (lib->functions != NULL) {
		int n = array_len (lib->functions) ;

		for (int i = 0; i < n; i++) {
			rm_free (lib->functions[i]) ;
		}
	}
	array_free (lib->functions) ;

	if (lib->script != NULL) {
		rm_free (lib->script) ;
	}

	rm_free (lib->name) ;
}

// initialize UDF repository
bool UDF_RepoInit(void) {
	ASSERT (udf_repo == NULL) ;

	udf_repo = rm_calloc (1, sizeof(UDF_Repository)) ;

	// repo read/write lock
	int res = pthread_rwlock_init (&udf_repo->rwlock, NULL) ;
	if (res) {
		return false ;
	}

	udf_repo->libs = array_new(UDF_Lib, 1) ;

	return (udf_repo->libs != NULL) ;
}

// return repo's version
UDF_RepoVersion UDF_RepoGetVersion(void) {
	ASSERT (udf_repo != NULL) ;

	return udf_repo->v ;
}

// populate the JSContext with registered libs
void UDF_RepoPopulateJSContext
(
	JSContext *js_ctx,  // context to populate
	UDF_RepoVersion *v  // [output] repo version
) {
	ASSERT (v        != NULL) ;
	ASSERT (js_ctx   != NULL) ;
	ASSERT (udf_repo != NULL) ;

	// set version
	*v = udf_repo->v ;

	// lock under READ
	pthread_rwlock_rdlock (&udf_repo->rwlock) ;

	// load each registered library
	int n = array_len (udf_repo->libs) ;
	for (int i = 0; i < n; i++) {
		const char *script = udf_repo->libs[i].script ;

		// evalute script
		JSValue val = JS_Eval (js_ctx, script, strlen (script), "<input>",
				JS_EVAL_TYPE_GLOBAL) ;

		ASSERT (!JS_IsException (val)) ;
		JS_FreeValue (js_ctx, val) ;
	}

	// unlock
	pthread_rwlock_unlock (&udf_repo->rwlock) ;
}

// returns number of registered libs
unsigned int UDF_RepoLibsCount(void) {
	ASSERT (udf_repo != NULL) ;

	return array_len (udf_repo->libs) ;
}

// get lib by name
bool UDF_RepoGetLib
(
	const char *name,         // lib's name
	const char ***functions,  // [optional] [output] lib's functions
	const char **script       // [optional] [output] lib's script
) {
	if (script    != NULL) *script    = NULL ;
	if (functions != NULL) *functions = NULL ;

	unsigned int idx;
	if (!UDF_RepoContainsLib (name, &idx)) {
		return false ;		
	}

	UDF_RepoGetLibIdx (idx, NULL, functions, script) ;

	return true ;
}

// get lib by index
void UDF_RepoGetLibIdx
(
	unsigned int idx,         // lib's index
	const char **name,        // [optional] [output] lib's name
	const char ***functions,  // [optional] [output] lib's functions
	const char **script       // [optional] [output] lib's script
) {
	ASSERT (idx < UDF_RepoLibsCount()) ;

	UDF_Lib *lib = udf_repo->libs + idx ;

	//--------------------------------------------------------------------------
	// set outputs
	//--------------------------------------------------------------------------

	if (name)      *name      = lib->name ;
	if (script)    *script    = lib->script ;
	if (functions) *functions = (const char**) lib->functions ;
}

// returns script from UDF repository
const char *UDF_RepoGetScript
(
	const char *lib  // UDF library
) {
	ASSERT (lib      != NULL) ;
	ASSERT (udf_repo != NULL) ;

	const char *script = NULL ;

	pthread_rwlock_rdlock (&udf_repo->rwlock) ;

	int n = array_len(udf_repo->libs) ;
	for (int i = 0; i < n; i++) {
		const char *_lib = udf_repo->libs[i].name ;

		if (strcmp (_lib, lib) == 0) {
			script = udf_repo->libs[i].script ;
			break ;
		}
	}

	pthread_rwlock_unlock (&udf_repo->rwlock) ;

	return script ;
}

// checks if UDF repository contains library
bool UDF_RepoContainsLib
(
	const char *lib,   // UDF library
	unsigned int *idx  // [optional] library index
) {
	ASSERT (lib      != NULL) ;
	ASSERT (udf_repo != NULL) ;

	int n = array_len (udf_repo->libs) ;

	for (int i = 0; i < n; i++) {
		if (strcmp (udf_repo->libs[i].name, lib) == 0) {
			if (idx != NULL) {
				*idx = i ;
			}	
			return true ;
		}
	}

	return false ;
}

// register a new UDF library
bool UDF_RepoRegisterLib
(
	const char *lib,    // library
	const char *script  // script
) {
	ASSERT (lib      != NULL) ;
	ASSERT (script   != NULL) ;
	ASSERT (udf_repo != NULL) ;

	if (UDF_RepoContainsLib (lib, NULL)) {
		return false ;
	}

	// lock under write
	pthread_rwlock_wrlock (&udf_repo->rwlock) ;

	UDF_Lib _lib = {.name   = rm_strdup (lib),
					.script = rm_strdup (script),
					.functions = NULL } ;

	array_append (udf_repo->libs, _lib) ;

	// unlock
	pthread_rwlock_unlock (&udf_repo->rwlock) ;

	return true ;
}

static UDF_Lib *_UDF_RepoGetLib
(
	const char *lib,   // library name
	unsigned int *idx  // [optional] [output] lib index
) {
	ASSERT (lib      != NULL) ;	
	ASSERT (udf_repo != NULL) ;	

	int n = array_len (udf_repo->libs) ;
	for (int i = 0; i < n; i++) {
		UDF_Lib *_lib = udf_repo->libs + i ;

		if (strcmp (lib, _lib->name) == 0) {
			if (idx != NULL) {
				*idx = i ;
			}

			return _lib ;
		}
	}

	return NULL ;
}

// register a new function for library
bool UDF_RepoRegisterFunc
(
	const char *lib,  // library
	const char *func  // function
) {
	ASSERT (lib  != NULL) ;
	ASSERT (func != NULL) ;
	
	UDF_Lib *_lib = _UDF_RepoGetLib (lib, NULL) ;
	ASSERT (_lib != NULL) ;

	if (unlikely (_lib->functions == NULL)) {
		_lib->functions = array_new (char *, 1) ;
	}

	// make sure function isn't already registered
	int n = array_len (_lib->functions) ;
	for (int i = 0; i < n; i++) {
		const char *_func = _lib->functions[i] ;
		if (strcmp (_func, func) == 0) {
			return false ;
		}
	}

	// new function
	array_append (_lib->functions, rm_strdup (func)) ;

	return true ;
}

// removes a UDF library from repository
bool UDF_RepoRemoveLib
(
	const char *lib,     // UDF library
	const char **script  // [optional] [output] removed script
) {
	ASSERT (lib      != NULL) ;
	ASSERT (udf_repo != NULL) ;
	
	// locate library
	unsigned int idx ;
	UDF_Lib *_lib = _UDF_RepoGetLib (lib, &idx) ;

	// return if library doesn't exists
	if (_lib == NULL) {
		return false ;
	}

	// backup script if required
	if (script != NULL) {
		*script = udf_repo->libs[idx].script ;
		udf_repo->libs[idx].script = NULL ;
	}

	// lock under WRITE
	pthread_rwlock_wrlock (&udf_repo->rwlock) ;

	// free library
	UDF_Lib_Free (_lib) ;

	// remove library from repo
	array_del_fast (udf_repo->libs, idx);

	// bump version
	udf_repo->v++ ;

	// unlock
	pthread_rwlock_unlock (&udf_repo->rwlock) ;

	return true ;
}

// expose library by:
// 1. bumping repository version (causing others to pick up the latest version)
// 2. introduce library's functions to the global UDF functions repo
void UDF_RepoExposeLib
(
	const char *lib  // library to expose
) {
	ASSERT (lib      != NULL) ;
	ASSERT (udf_repo != NULL) ;

	UDF_Lib *_lib = _UDF_RepoGetLib (lib, NULL) ;
	ASSERT (_lib != NULL) ;

	// bump version
	udf_repo->v++ ;

	int n = array_len (_lib->functions) ;
	for (int i = 0; i < n ; i++) {
		const char *func_name = _lib->functions[i] ;
		AR_FuncRegisterUDF (func_name) ;
	}
}

// free UDF repository
void UDF_RepoFree(void) {
	if (udf_repo == NULL) {
		return ;
	}

	// free each registered library
	for (int i = 0; i < array_len (udf_repo->libs); i++) {
		UDF_Lib * lib = udf_repo->libs + i ;
		UDF_Lib_Free (lib) ;
	}

	array_free (udf_repo->libs) ;

	// free lock
	int res = pthread_rwlock_destroy (&udf_repo->rwlock) ;
	ASSERT (res == 0) ;

	rm_free (udf_repo) ;
}

