/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "GraphBLAS.h"

// runs contraction hierarchies over 'A'. on success:
//   *S_out    - GrB_FP64, same dims as A: every shortcut edge inserted
//               during the run, keyed by its final (minimum) weight.
//               caller owns it and must GrB_free it.
//   *rank_out - GrB_INT64, same dims as A: 1-based contraction rank per
//               node. entries exist ONLY for nodes that were actually
//               contracted -- isolated (degree-0 within A) nodes have no
//               entry. caller owns it and must GrB_free it.
void ContractionHierarchies_Contract
(
	GrB_Matrix  A,         // input weight matrix
	GrB_Matrix *S_out,     // [output] shortcut weight overlay
	GrB_Vector *rank_out   // [output] per-node contraction rank
) ;
