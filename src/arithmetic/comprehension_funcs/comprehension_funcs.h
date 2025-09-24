/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../arithmetic_expression.h"
#include "../../filter_tree/filter_tree.h"

typedef struct {
	const char *variable_str;  // String name of the comprehension's local variable
	int variable_idx;          // Record index of the comprehension's local variable
	FT_FilterNode *ft;         // [optional] The predicate tree to evaluate each element against.
	AR_ExpNode *eval_exp;      // [optional] The projection routine to build each return element.
	Record local_record;       // Record to populate with the input record's values and the list element.
} ListComprehensionCtx;

void Register_ComprehensionFuncs(void);

