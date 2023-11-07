//------------------------------------------------------------------------------
// GB_ewise_fulla_template: C += A+B where all 3 matrices are dense
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// No matrix is iso.  This template is not used for any generic kernels.

{

    //--------------------------------------------------------------------------
    // get A, B, and C
    //--------------------------------------------------------------------------

    // any matrix may be aliased to any other (C==A, C==B, and/or A==B)
    GB_A_TYPE *Ax = (GB_A_TYPE *) A->x ;
    GB_B_TYPE *Bx = (GB_B_TYPE *) B->x ;
    GB_C_TYPE *Cx = (GB_C_TYPE *) C->x ;
    GB_C_NVALS (cnz) ;      // const int64_t cnz = GB_nnz (C) ;
    ASSERT (!C->iso) ;
    ASSERT (!A->iso) ;
    ASSERT (!B->iso) ;
    int64_t p ;

    //--------------------------------------------------------------------------
    // C += A+B where all 3 matries are dense
    //--------------------------------------------------------------------------

    if (A_is_B)
    {

        //----------------------------------------------------------------------
        // C += A+A where A and C are dense
        //----------------------------------------------------------------------

        // C += A+A
        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (p = 0 ; p < cnz ; p++)
        { 
            #if GB_OP_IS_SECOND
            GB_DECLAREB (aij) ;
            GB_GETB (aij, Ax, p, false) ;           // aij = Ax [p]
            #else
            GB_DECLAREA (aij) ;
            GB_GETA (aij, Ax, p, false) ;           // aij = Ax [p]
            #endif
            GB_C_TYPE z1 ;                          // declare scalar z1
            GB_BINOP (z1, aij, aij, 0, 0) ;         // z1 = aij + aij
            GB_C_TYPE z0 = Cx [p] ;                 // z0 = cij
            GB_EWISEOP (Cx, p, z0, z1, 0, 0) ;      // Cx [p] = z0 + z1
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // C += A+B where all 3 matrices are dense
        //----------------------------------------------------------------------

        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (p = 0 ; p < cnz ; p++)
        { 
            GB_DECLAREA (aij) ;
            GB_GETA (aij, Ax, p, false) ;           // aij = Ax [p]
            GB_DECLAREB (bij) ;
            GB_GETB (bij, Bx, p, false) ;           // bij = Bx [p]
            GB_C_TYPE z1 ;                          // declare scalar z1
            GB_BINOP (z1, aij, bij, 0, 0) ;         // z1 = aij + bij
            GB_C_TYPE z0 = Cx [p] ;                 // z0 = cij
            GB_EWISEOP (Cx, p, z0, z1, 0, 0) ;      // Cx [p] = z0 + z1
        }
    }
}

