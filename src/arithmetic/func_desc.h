/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include <sys/types.h>
#include "../value.h"

#define VAR_ARG_LEN UINT_MAX

// an aggregation context
// each aggregation function operates on an aggregation context
// this is where the function can maintin its internal state
typedef struct {
	SIValue result;     // aggregation value
	void *private_data; // [optional] addition private data
} AggregateCtx;

// AR_Func - function pointer to an operation with an arithmetic expression
typedef SIValue(*AR_Func)(SIValue *argv, int argc, void *private_data);

// AR_Func_Finalize - function pointer to a routine for computing an aggregate function's final value
typedef void (*AR_Func_Finalize)(void *ctx);

// AR_Func_Free - function pointer to a routine for freeing a function's private data
typedef void (*AR_Func_Free)(void *ctx);

// AR_Func_Clone - function pointer to a routine for cloning a function's private data
typedef void *(*AR_Func_Clone)(void *orig);

// AR_Func_PrivateData - function pointer to a routine which produce function's private data
typedef AggregateCtx *(*AR_Func_PrivateData)(void);

// aggregation function callbacks
typedef struct {
	AR_Func_Free free;                  // [optional] function pointer to cleanup routine
	AR_Func_Clone clone;                // [optional] function pointer to clone routine
	AR_Func_Finalize finalize;          // [optional] function pointer to finalizing aggregate value routine
	AR_Func_PrivateData private_data;   // function pointer to private data generator
} AR_FuncCBs;

typedef struct {
	AR_Func func;          // function pointer to scalar or aggregate function routine
	SIType *types;         // types of arguments
	SIType ret_type;       // return type
	uint min_argc;         // minimal number of arguments function expects
	uint max_argc;         // maximal number of arguments function expects
	bool internal;         // is function internal
	bool reducible;        // can be reduced using static evaluation
	bool aggregate;        // true if the function is an aggregation
	bool deterministic;    // true if return value is predictable
	const char *name;      // function name
	AR_FuncCBs callbacks;  // aggregation callbacks
} AR_FuncDesc;

// create a new function descriptor
AR_FuncDesc *AR_FuncDescNew
(
	const char *name,   // function name
	AR_Func func,       // pointer to function routine
	uint min_argc,      // minimal number of arguments
	uint max_argc,      // maximal number of arguments
	SIType *types,      // types of arguments
	SIType ret_type,    // return type
	bool internal,      // is function internal
	bool reducible,     // true if function is reducible
	bool deterministic  // true if return value is predictable
);

// register arithmetic function to repository
void AR_RegFunc
(
	AR_FuncDesc *func
);

// set the function pointers for cloning and freeing a function's private data
void AR_SetPrivateDataRoutines
(
	AR_FuncDesc *func_desc,
	AR_Func_Free free,
	AR_Func_Clone clone
);

// retrieves an arithmetic function by its name
AR_FuncDesc *AR_GetFunc
(
	const char *func_name,
	bool include_internal
);

// get function return type
SIType AR_FuncDesc_RetType
(
	const AR_FuncDesc *func	
);

// check to see if function exists
// TODO: move this function to more appropriate place
bool AR_FuncExists
(
	const char *func_name
);

// check to see if function is an aggregation
bool AR_FuncIsAggregate
(
	const char *func_name
);

//TODO: implement
void AR_FuncFree
(
	AR_FuncDesc *f
);

