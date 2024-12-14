/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "GraphBLAS.h"

// compute in/out degree for all nodes
//
// arguments:
// 1. 'label of nodes to compute degree for [optional] default to all
// 2. 'relationship-type' to consider [optional] default to all
// 3. 'dir' in/out/both [optiona] default to out
GrB_Info Degree
(
	GrB_Vector *degree,  // [output] degree vector
	GrB_Matrix A         // graph matrix	
);

