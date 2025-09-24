/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../op.h"
#include "../../../arithmetic/algebraic_expression.h"

void TraversalToString
(
	const OpBase *op,
	sds *buf,
	const AlgebraicExpression *ae
);

void ScanToString
(
	const OpBase *op,
	sds *buf,
	const char *alias,
	const char *label
);

