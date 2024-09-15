/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "entities/edge.h"

// checks if x represents scalar entry, if not x is a vector
#define SCALAR_ENTRY(x) !((x) & MSB_MASK)

typedef struct Tensor *Tensor;

// init new tensor
Tensor Tensor_new
(
	GrB_Index nrows,  // # rows
	GrB_Index ncols   // # columns
);

// set entry at T[row, col] = x
void Tensor_SetElement
(
	Tensor T,       // tensor
	GrB_Index row,  // row
	GrB_Index col,  // col
	uint64_t x      // value
);

// set multiple entries
void Tensor_SetElements
(
	Tensor T,           // tensor
	const Edge **edges  // assume edges are sorted by src and dest
);

// checks to see if tensor has pending changes
bool Tensor_pending
(
	const Tensor T   // tensor
);

// free tensor
void Tensor_free
(
	Tensor *T  // tensor
);

