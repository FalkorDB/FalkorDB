//------------------------------------------------------------------------------
// LG_BreadthFirstSearch_SSGrB:  BFS using Suitesparse extensions
//------------------------------------------------------------------------------

// LAGraph, (c) 2021 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
//
// See additional acknowledgments in the LICENSE file,
// or contact permission@sei.cmu.edu for the full terms.

//------------------------------------------------------------------------------

// References:
//
// Carl Yang, Aydin Buluc, and John D. Owens. 2018. Implementing Push-Pull
// Efficiently in GraphBLAS. In Proceedings of the 47th International
// Conference on Parallel Processing (ICPP 2018). ACM, New York, NY, USA,
// Article 89, 11 pages. DOI: https://doi.org/10.1145/3225058.3225122
//
// Scott Beamer, Krste Asanovic and David A. Patterson, The GAP Benchmark
// Suite, http://arxiv.org/abs/1508.03619, 2015.  http://gap.cs.berkeley.edu/

#pragma once

#include "GraphBLAS.h"

//****************************************************************************

int LAGr_BreadthFirstSearch_MaxLevel
(
	GrB_Vector    *level,
	GrB_Vector    *parent,
	GrB_Matrix    A,
	GrB_Index     src,
	GrB_Index     *dest,
	GrB_Index     max_level
) ;

