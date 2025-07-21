/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#include "func_desc.h"
#include "../RG.h"
#include "../util/rmalloc.h"
#include "../util/strutil.h"
#include "../../deps/rax/rax.h"
#include "aggregate_funcs/agg_funcs.h"
#include <ctype.h>

// Arithmetic function repository
rax *__aeRegisteredFuncs = NULL;

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
	AR_FuncDesc *desc = rm_calloc(1, sizeof(AR_FuncDesc));

	desc->name                    =  name;
	desc->func                    =  func;
	desc->types                   =  types;
	desc->ret_type                =  ret_type;
	desc->min_argc                =  min_argc;
	desc->max_argc                =  max_argc;
	desc->internal                =  internal;
	desc->aggregate               =  false;
	desc->reducible               =  reducible;

	return desc;
}

// register an arithmetic function
void AR_RegFunc
(
	AR_FuncDesc *func
) {
	char lower_func_name[32];
	size_t lower_func_name_len = 32;
	str_tolower_ascii(func->name, lower_func_name, &lower_func_name_len);
	int res = raxInsert(__aeRegisteredFuncs, (unsigned char *)lower_func_name,
			lower_func_name_len, func, NULL);
	ASSERT(res == 1);
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
	const char *func_name,
	bool include_internal
) {
	size_t len = strlen(func_name);
	char lower_func_name[len + 1];
	str_tolower_ascii(func_name, lower_func_name, &len);
	void *f = raxFind(__aeRegisteredFuncs, (unsigned char *)lower_func_name, len);

	if(f == raxNotFound) return NULL;

	AR_FuncDesc *func = (AR_FuncDesc*)f;

	if(func->internal && !include_internal) return NULL;

	return func;
}

SIType AR_FuncDesc_RetType
(
	const AR_FuncDesc *func	
) {
	ASSERT(func != NULL);

	return func->ret_type;
}

bool AR_FuncExists
(
	const char *func_name
) {
	size_t len = strlen(func_name);
	char lower_func_name[len + 1];
	str_tolower_ascii(func_name, lower_func_name, &len);
	void *f = raxFind(__aeRegisteredFuncs, (unsigned char *)lower_func_name, len);

	if(f == raxNotFound) return false;

	AR_FuncDesc *func = (AR_FuncDesc*)f;

	return !func->internal;
}

bool AR_FuncIsAggregate
(
	const char *func_name
) {
	size_t len = strlen(func_name);
	char lower_func_name[len + 1];
	str_tolower_ascii(func_name, lower_func_name, &len);
	AR_FuncDesc *f = raxFind(__aeRegisteredFuncs,
			(unsigned char *)lower_func_name, len);

	if(f == raxNotFound) return false;

	return f->aggregate;
}

