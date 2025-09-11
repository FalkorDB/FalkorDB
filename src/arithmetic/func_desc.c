/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "func_desc.h"
#include "../util/arr.h"
#include "../util/rmalloc.h"
#include "../util/strutil.h"
#include "aggregate_funcs/agg_funcs.h"

#include <ctype.h>
#include <pthread.h>

#define FUNC_NAME_MAX_LEN 64

typedef struct {
	rax *repo;
	pthread_rwlock_t lock;
} UDFS ;

UDFS *__udfs              = NULL ;
rax  *__aeRegisteredFuncs = NULL ;

static void _NormalizeFunctionName
(
	const char *name,
	char normalized[FUNC_NAME_MAX_LEN],
	size_t *len
) {
	ASSERT (len  != NULL) ;
	ASSERT (name != NULL) ;

	*len = FUNC_NAME_MAX_LEN ;

	// convert function name to lowercase
	str_tolower_ascii (name, normalized, len) ;
}

void AR_InitFuncsRepo(void) {
	ASSERT (__aeRegisteredFuncs == NULL) ;

	__aeRegisteredFuncs = raxNew () ;

	//--------------------------------------------------------------------------
	// initialize User Defined Functions AR repo
	//--------------------------------------------------------------------------

	__udfs = rm_calloc (1, sizeof (UDFS)) ;
	pthread_rwlock_init (&__udfs->lock, NULL) ;
	__udfs->repo = raxNew () ;
}

void AR_FinalizeFuncsRepo(void) {
	ASSERT (__udfs              != NULL) ;
	ASSERT (__aeRegisteredFuncs != NULL) ;

	//--------------------------------------------------------------------------
	// free each registered function
	//--------------------------------------------------------------------------

	raxIterator it;
	raxStart (&it, __aeRegisteredFuncs) ;

	// retrieve the first key in the rax
	raxSeek (&it, "^", NULL, 0);
	while (raxNext (&it)) {
		AR_FuncDesc *f = it.data ;
		AR_FuncFree (f) ;
	}

	raxStop (&it);
	raxFree (__aeRegisteredFuncs) ;

	//--------------------------------------------------------------------------
	// free UDFs
	//--------------------------------------------------------------------------

	raxStart (&it, __udfs->repo) ;

	// retrieve the first key in the rax
	raxSeek (&it, "^", NULL, 0);
	while (raxNext (&it)) {
		AR_FuncDesc *f = it.data ;
		rm_free ((void*)f->name) ;
		AR_FuncFree (f) ;
	}

	pthread_rwlock_destroy (&__udfs->lock) ;

	raxStop (&it);
	raxFree (__udfs->repo) ;
}

AR_FuncDesc *AR_FuncDescNew
(
	const char *name,
	AR_Func func,
	uint min_argc,
	uint max_argc,
	SIType *types,
	SIType ret_type,
	bool internal,
	bool reducible
) {
	AR_FuncDesc *desc = rm_calloc (1, sizeof(AR_FuncDesc)) ;

	desc->name      = name ;
	desc->func      = func ;
	desc->types     = types ;
	desc->ret_type  = ret_type ;
	desc->min_argc  = min_argc ;
	desc->max_argc  = max_argc ;
	desc->internal  = internal ;
	desc->aggregate = false ;
	desc->reducible = reducible ;

	return desc ;
}

// register function to repository
void AR_FuncRegister
(
	AR_FuncDesc *func  // function to register
) {
	ASSERT (func                != NULL) ;
	ASSERT (__aeRegisteredFuncs != NULL) ;

	size_t len;
	char   lower_func_name[FUNC_NAME_MAX_LEN];
	_NormalizeFunctionName (func->name, lower_func_name, &len) ;

	// add function to repository
	int res = raxInsert(__aeRegisteredFuncs, (unsigned char *)lower_func_name,
			len, func, NULL);
	ASSERT(res == 1);
}

// forward declaration
SIValue AR_UDF (SIValue *argv, int argc, void *private_data) ;

void AR_FuncRegisterUDF
(
	const char *name
) {
	SIType ret_type = SI_ALL ;
	SIType *types = array_new (SIType, 2) ;
	array_append (types, T_STRING) ;
	array_append (types, SI_ALL) ;

	AR_FuncDesc *func = AR_FuncDescNew (rm_strdup (name), AR_UDF, 1,
			VAR_ARG_LEN, types, ret_type, false, false) ;

	func->udf = true ;

	// WRITE lock
	int res = pthread_rwlock_wrlock (&__udfs->lock) ;
	ASSERT (res == 0) ;

	// add UDF to repo
	res = raxInsert(__udfs->repo, (unsigned char *)name, strlen(name), func,
			NULL);
	ASSERT(res == 1);

	// unlock
	res = pthread_rwlock_unlock (&__udfs->lock) ;
	ASSERT (res == 0) ;
}

// unregister function to repository
bool AR_FuncRemoveUDF
(
	const char *func_name  // function name to remove from repository
) {
	ASSERT (func_name != NULL) ;

	// WRITE lock
	int res = pthread_rwlock_wrlock (&__udfs->lock) ;
	ASSERT (res == 0) ;

	// remove function from repository
	AR_FuncDesc *func = NULL ;
	int removed = raxRemove(__udfs->repo, (unsigned char *)func_name,
			strlen (func_name), (void**)func) ;

	// TODO: leaking func!

	// unlock
	res = pthread_rwlock_unlock (&__udfs->lock) ;
	ASSERT (res == 0) ;

	return removed == 1 ;
}

inline void AR_SetPrivateDataRoutines
(
	AR_FuncDesc *func_desc,
	AR_Func_Free free,
	AR_Func_Clone clone
) {
	func_desc->callbacks.free = free;
	func_desc->callbacks.clone = clone;
}

// get arithmetic function
AR_FuncDesc *AR_GetFunc
(
	const char *func_name,  // function to lookup
	bool include_internal   // alow using internal functions
) {
	ASSERT (func_name           != NULL) ;
	ASSERT (__udfs              != NULL) ;
	ASSERT (__aeRegisteredFuncs != NULL) ;

	// normalize function name by lowercasing
	size_t len = strlen (func_name) ;
	char lower_func_name[len + 1] ;
	str_tolower_ascii (func_name, lower_func_name, &len) ;

	// lookup function
	void *f = raxFind (__aeRegisteredFuncs,
			(unsigned char *)lower_func_name, len) ;

	// native function wasn't found, search UDFs
	if (f == raxNotFound) {
		//----------------------------------------------------------------------
		// search UDFs
		//----------------------------------------------------------------------

		// READ lock
		int res = pthread_rwlock_rdlock (&__udfs->lock) ;
		ASSERT (res == 0) ;

		// search UDFs repo
		f = raxFind (__udfs->repo, (unsigned char *)func_name, len) ;

		// unlock
		res = pthread_rwlock_unlock (&__udfs->lock) ;
		ASSERT (res == 0) ;

		if (f == raxNotFound) {
			return NULL ;
		}
	}

	AR_FuncDesc *func = (AR_FuncDesc*)f ;

	if (func->internal && !include_internal) {
		return NULL ;
	}

	return func ;
}

SIType AR_FuncDesc_RetType
(
	const AR_FuncDesc *func	
) {
	ASSERT(func != NULL);

	return func->ret_type;
}

// returns true if function is in repository
bool AR_FuncExists
(
	const char *func_name  // function name to lookup
) {
	ASSERT (func_name           != NULL) ;
	ASSERT (__udfs              != NULL) ;
	ASSERT (__aeRegisteredFuncs != NULL) ;

	return (AR_GetFunc (func_name, false) != NULL) ;
}

// returns true if function is an aggregation function
bool AR_FuncIsAggregate
(
	const char *func_name  // function name
) {
	ASSERT (func_name           != NULL) ;
	ASSERT (__aeRegisteredFuncs != NULL) ;

	AR_FuncDesc *f = AR_GetFunc ( func_name,  true);

	if(f == NULL) {
		return false ;
	}

	return f->aggregate ;
}

void AR_FuncFree
(
	AR_FuncDesc *f
) {
	ASSERT (f != NULL) ;

	rm_free (f) ;
}

