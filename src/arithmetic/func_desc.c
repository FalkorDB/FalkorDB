/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2) or the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "func_desc.h"
#include "builtin_funcs_lookup.h"
#include "../util/arr.h"
#include "../util/rmalloc.h"
#include "../util/strutil.h"
#include "aggregate_funcs/agg_funcs.h"

#include <ctype.h>
#include <pthread.h>

typedef struct {
	rax *repo;
	pthread_rwlock_t lock;
} UDFS;

UDFS *__udfs = NULL ;

static void _NormalizeFunctionName
(
	char *name,
	size_t *len
) {
	ASSERT (len  != NULL) ;
	ASSERT (name != NULL) ;

	*len = strlen (name) ;
	str_tolower_ascii (name, name, len) ;
}

void AR_InitFuncsRepo(void) {
	// Built-in functions live in static storage (builtin_funcs.c).
	// Only the UDF repo requires dynamic initialisation.
	__udfs = rm_calloc (1, sizeof (UDFS)) ;
	pthread_rwlock_init (&__udfs->lock, NULL) ;
	__udfs->repo = raxNew () ;
}

void AR_FinalizeFuncsRepo(void) {
	ASSERT (__udfs != NULL) ;

	raxIterator it ;
	raxStart (&it, __udfs->repo) ;
	raxSeek (&it, "^", NULL, 0) ;
	while (raxNext (&it)) {
		AR_FuncDesc *f = it.data ;
		free ((void *)f->name) ;
		AR_FuncFree (f) ;
	}
	raxStop (&it) ;
	raxFree (__udfs->repo) ;

	pthread_rwlock_destroy (&__udfs->lock) ;
	rm_free (__udfs) ;
	__udfs = NULL ;
}

// create a new function descriptor
AR_FuncDesc *AR_FuncDescNew
(
	char *name,         // function name
	AR_Func func,       // pointer to function routine
	uint8_t min_argc,   // minimal number of arguments
	uint8_t max_argc,   // maximal number of arguments
	SIType *types,      // types of arguments
	SIType ret_type,    // return type
	bool internal,      // is function internal
	bool reducible,     // true if function is reducible
	bool deterministic  // true if return value is predictable
) {
	AR_FuncDesc *desc = rm_calloc(1, sizeof(AR_FuncDesc));

	desc->name          = name;
	desc->func          = func;
	desc->types         = types;
	desc->types_len     = (uint8_t)arr_len (types) ;
	desc->ret_type      = ret_type;
	desc->min_argc      = min_argc;
	desc->max_argc      = max_argc;
	desc->internal      = internal;
	desc->reducible     = reducible;
	desc->deterministic = deterministic;

	return desc;
}

// forward declaration
SIValue AR_UDF (SIValue *argv, int argc, void *private_data) ;

// register a new UDF function
void AR_FuncRegisterUDF
(
	char *name  // full name (lib.func)
) {
	ASSERT (name != NULL) ;

	size_t len ;
	_NormalizeFunctionName (name, &len) ;

	SIType  ret_type = SI_ALL ;
	SIType *types    = arr_new (SIType, 3) ;
	arr_append (types, T_STRING) ;
	arr_append (types, T_STRING) ;
	arr_append (types, SI_ALL) ;

	AR_FuncDesc *func = AR_FuncDescNew (name, AR_UDF, 2, VAR_ARG_LEN,
			types, ret_type, false, false, false) ;
	func->udf = true ;

	// WRITE lock
	int res = pthread_rwlock_wrlock (&__udfs->lock) ;
	ASSERT (res == 0) ;

	res = raxInsert (__udfs->repo, (unsigned char *)name, len, func, NULL) ;
	ASSERT (res == 1) ;

	res = pthread_rwlock_unlock (&__udfs->lock) ;
	ASSERT (res == 0) ;
}

// unregister a UDF function
bool AR_FuncRemoveUDF
(
	char *func_name
) {
	ASSERT (func_name != NULL) ;

	size_t len ;
	_NormalizeFunctionName (func_name, &len) ;

	int res = pthread_rwlock_wrlock (&__udfs->lock) ;
	ASSERT (res == 0) ;

	AR_FuncDesc *func = NULL ;
	int removed = raxRemove (__udfs->repo, (unsigned char *)func_name, len,
			(void **)&func) ;
	ASSERT (removed == 1) ;

	free (func->name) ;

	res = pthread_rwlock_unlock (&__udfs->lock) ;
	ASSERT (res == 0) ;

	return removed == 1 ;
}

inline void AR_SetPrivateDataRoutines
(
	AR_FuncDesc *func_desc,
	AR_Func_Free free,
	AR_Func_Clone clone,
	AR_Func_PrivateDataAliases aliases
) {
	ASSERT (func_desc->callbacks.free    == NULL) ;
	ASSERT (func_desc->callbacks.clone   == NULL) ;
	ASSERT (func_desc->callbacks.aliases == NULL) ;

	func_desc->callbacks.free    = free ;
	func_desc->callbacks.clone   = clone ;
	func_desc->callbacks.aliases = aliases ;
}

// get arithmetic function
AR_FuncDesc *AR_GetFunc
(
	const char *func_name,  // function to lookup
	bool include_internal   // allow using internal functions
) {
	ASSERT (func_name != NULL) ;
	ASSERT (__udfs    != NULL) ;

	// normalize to lowercase
	size_t len = strlen (func_name) ;
	char   lower[len + 1] ;
	str_tolower_ascii (func_name, lower, &len) ;

	// look up in the static built-in table via perfect hash
	AR_FuncDesc *f = AR_BuiltinFuncLookup (lower, len) ;
	if (f == NULL) {
		// search UDFs
		int res = pthread_rwlock_rdlock (&__udfs->lock) ;
		ASSERT (res == 0) ;

		void *p = raxFind (__udfs->repo, (unsigned char *)lower, len) ;

		res = pthread_rwlock_unlock (&__udfs->lock) ;
		ASSERT (res == 0) ;

		if (p == raxNotFound) {
			return NULL ;
		}

		f = (AR_FuncDesc *)p ;
	}

	if (f->internal && !include_internal) {
		return NULL ;
	}
	return f ;
}

SIType AR_FuncDesc_RetType
(
	const AR_FuncDesc *func
) {
	ASSERT(func != NULL);
	return func->ret_type;
}

// returns true if function is in the repository
bool AR_FuncExists
(
	const char *func_name
) {
	ASSERT (func_name != NULL) ;
	ASSERT (__udfs    != NULL) ;

	return AR_GetFunc (func_name, false) != NULL ;
}

// returns true if function is an aggregation function
bool AR_FuncIsAggregate
(
	const char *func_name
) {
	ASSERT (func_name != NULL) ;

	AR_FuncDesc *f = AR_GetFunc (func_name, true) ;
	return (f != NULL && f->aggregate) ;
}

// free a heap-allocated function descriptor (UDFs only;
// never call on the static built-in descriptors)
void AR_FuncFree
(
	AR_FuncDesc *f
) {
	ASSERT(f != NULL);

	if (f->types != NULL) {
		arr_free (f->types) ;
	}
	rm_free (f) ;
}
