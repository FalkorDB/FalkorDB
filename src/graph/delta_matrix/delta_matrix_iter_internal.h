/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

// private helpers shared between delta_matrix_iter.c and
// delta_matrix_iter_sorted.c - not part of the public API, do not include
// this header outside this directory

#include "./delta_matrix_iter.h"

// returns true if iterator is detached from a matrix
#define IS_DETACHED(iter) ((iter) == NULL || (iter)->A == NULL)

// advance a plain row iterator, updating `depleted`
void _Iter_next
(
	GxB_Iterator it,
	GrB_Index max_row,
	bool *depleted
) ;

// advance mi to its next live entry (or depleted) - skips any entry masked
// out by a pending deletion, so mi->it's position is always live whenever
// mi->depleted is false
void _MaskedMIter_next
(
	Delta_MaskedMIter *mi,
	GrB_Index max_row
) ;
