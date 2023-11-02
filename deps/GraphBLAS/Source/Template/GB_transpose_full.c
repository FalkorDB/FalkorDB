//------------------------------------------------------------------------------
// GB_transpose_full: C=op(cast(A')), transpose, typecast, and apply op
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

{

    //------------------------------------------------------------------
    // A and C are both full
    //------------------------------------------------------------------

    ASSERT (Workspaces == NULL) ;
    ASSERT (GB_IS_FULL (A)) ;
    ASSERT (GB_IS_FULL (C)) ;

    // A is avlen-by-avdim; C is avdim-by-avlen
    int64_t avlen = A->vlen ;
    int64_t avdim = A->vdim ;
    int64_t anz = avlen * avdim ;   // ignore integer overflow

    // TODO: it would be faster to do this by tiles, not rows/columns,
    // for large matrices, but in most of the cases in GraphBLAS, A and
    // C will be tall-and-thin or short-and-fat.

    #ifndef GB_ISO_TRANSPOSE
    int tid ;
    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (tid = 0 ; tid < nthreads ; tid++)
    {
        int64_t pC_start, pC_end ;
        GB_PARTITION (pC_start, pC_end, anz, tid, nthreads) ;
        for (int64_t pC = pC_start ; pC < pC_end ; pC++)
        { 
            // get i and j of the entry C(i,j)
            // i = (pC % avdim) ;
            // j = (pC / avdim) ;
            // find the position of the entry A(j,i) 
            // pA = j + i * avlen
            // Cx [pC] = op (Ax [pA])
            GB_APPLY_OP (pC, ((pC/avdim) + (pC%avdim) * avlen)) ;
        }
    }
    #endif
}

