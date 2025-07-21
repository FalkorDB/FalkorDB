/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#include "arithmetic_op.h"

AST_Operator ArithmeticOp_ReverseOp(AST_Operator op) {
	switch(op) {
	case OP_LT:
		return OP_GT;
	case OP_LE:
		return OP_GE;
	case OP_GT:
		return OP_LT;
	case OP_GE:
		return OP_LE;
	default:
		return op;
	}
}
