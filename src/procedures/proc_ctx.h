/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../value.h"

// Procedure accepts a variable number of arguments.
#define PROCEDURE_VARIABLE_ARG_COUNT UINT_MAX

// Procedure response type.
typedef enum {
	PROCEDURE_OK = 0,
	PROCEDURE_ERR = (1 << 0),
} ProcedureResult;

// Procedure internal state.
typedef enum {
	PROCEDURE_NOT_INIT = 0,         // Start state.
	PROCEDURE_INIT = (1 << 0),      // Once invoked is called.
	PROCEDURE_DEPLETED = (1 << 1),  // Once step can no longer produce data.
	PROCEDURE_ERROR = (1 << 2),     // Whenever an error occurred.
} ProcedureState;

// Procedure output
typedef struct {
	char *name;     // Name of output.
	SIType type;    // Type of output.
} ProcedureOutput;

struct ProcedureCtx;

// Procedure instance generator.
typedef struct ProcedureCtx *(*ProcGenerator)(void);
// Procedure step function.
typedef SIValue *(*ProcStep)(struct ProcedureCtx *ctx);
// Procedure function pointer.
typedef ProcedureResult(*ProcInvoke)(struct ProcedureCtx *ctx, const SIValue *args, const char **yield);
// Procedure free resources.
typedef ProcedureResult(*ProcFree)(struct ProcedureCtx *ctx);

/* ProcedureCtx */
struct ProcedureCtx {
	const char *name;           // Procedure name.
	ProcedureState state;       // State in which the procedure is in.
	unsigned int argc;          // Number of arguments procedure accepts.
	ProcedureOutput *output;    // Procedure possible output(s).
	void *privateData;          //
	ProcStep Step;              //
	ProcInvoke Invoke;          //
	ProcFree Free;              //
	bool readOnly;              // Indicates if the procedure is able to mutate the graph.
};
typedef struct ProcedureCtx ProcedureCtx;

ProcedureCtx *ProcCtxNew(
	const char *name,           // Procedure name.
	unsigned int argc,          // Procedure arguments.
	ProcedureOutput *output,    // Procedure output.
	ProcStep fStep,             // Procedure Step function.
	ProcInvoke fInvoke,         // Procedure Invoke function.
	ProcFree fFree,             // Procedure Free function.
	void *privateData,          // Procedure private data.
	bool readOnly               // Indicates if the procedure is able to mutate the graph.
);

