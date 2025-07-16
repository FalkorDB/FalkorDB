/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#pragma once

#include "../../filter_tree/filter_tree.h"
#include "../../arithmetic/algebraic_expression.h"
#include "../../../deps/rax/rax.h"

// algebraic expression associated with a score
typedef struct {
	int score;                 // score given to expression
	AlgebraicExpression *exp;  // algebraic expression
} ScoredExp;

 // collect independent entities
 // and the number of their independent occurrences from a filter tree
 // an indpendent entity is an entity that is the single entity in a predicate
 // 'root' - filter tree root
 // 'entities' rax holding independent entities frequency
void FilterTree_CollectIndependentEntities
(
	const FT_FilterNode *root,
	rax *entities
);

// associates each expression with a score
// a score for a given expression is based on a score given to other expressions
void TraverseOrder_ScoreExpressions
(
	ScoredExp *scored_exps,      // sorted array of scored expressions
	AlgebraicExpression **exps,  // expressions to score
	uint nexp,                   // number of expressions
	rax *bound_vars,             // map of bounded entities
	rax *filtered_entities,      // map of filtered entities
	const QueryGraph *qg         // query graph
);

