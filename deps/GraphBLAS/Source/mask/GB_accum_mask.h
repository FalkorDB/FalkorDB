//------------------------------------------------------------------------------
// GB_accum_mask.h: definitions for GB_accum_mask
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_ACCUM_MASK_H
#define GB_ACCUM_MASK_H

#include "GB.h"

GrB_Info GB_accum_mask          // C<M> = accum (C,T)
(
    GrB_Matrix C,               // input/output matrix for results
    const GrB_Matrix M_in,      // optional mask for C, unused if NULL
    const GrB_Matrix MT_in,     // MT=M' if computed already in the caller
    const GrB_BinaryOp accum,   // optional accum for Z=accum(C,results)
    GrB_Matrix *Thandle,        // results of computation, freed when done
    const bool C_replace,       // if true, clear C first
    const bool Mask_comp,       // if true, complement the mask
    const bool Mask_struct,     // if true, use the only structure of M
    GB_Werk Werk
) ;

#endif

