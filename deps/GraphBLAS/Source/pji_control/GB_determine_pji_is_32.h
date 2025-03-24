//------------------------------------------------------------------------------
// GB_determine_pji_is_32: determine [pji]_is_32 for a new matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The Werk->[pji]_control has been determined from the input matrix and global
// controls.  If Werk is NULL, then only the global controls are used.  This
// method then determines the final p_is_32, j_is_32, and i_is_32 for a new
// matrix of the requested size.

#ifndef GB_DETERMINE_PJI_IS_32
#define GB_DETERMINE_PJI_IS_32

static inline void GB_determine_pji_is_32
(
    // output
    bool *p_is_32,      // if true, Ap will be 32 bits; else 64
    bool *j_is_32,      // if true, Ah and A->Y will be 32 bits; else 64
    bool *i_is_32,      // if true, Ai will be 32 bits; else 64
    // input
    int sparsity,       // sparse, hyper, bitmap, full, or auto (sparse/hyper)
    int64_t nvals,      // upper bound on # of entries in the matrix to create
    int64_t vlen,       // dimensions of the matrix to create
    int64_t vdim,
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (p_is_32 != NULL) ;
    ASSERT (j_is_32 != NULL) ;
    ASSERT (i_is_32 != NULL) ;

    //--------------------------------------------------------------------------
    // determine the 32/64 bit integer sizes for a new matrix
    //--------------------------------------------------------------------------

    if (sparsity == GxB_FULL || sparsity == GxB_BITMAP)
    {
    
        //----------------------------------------------------------------------
        // full/bitmap matrices do not have any integer sizes
        //----------------------------------------------------------------------

        (*p_is_32) = false ;
        (*j_is_32) = false ;
        (*i_is_32) = false ;

    }
    else
    {

        //----------------------------------------------------------------------
        // determine the 32/64 integer sizes for a sparse/hypersparse matrix
        //----------------------------------------------------------------------

        int8_t p_control = Werk ? Werk->p_control : GB_Global_p_control_get ( );
        int8_t j_control = Werk ? Werk->j_control : GB_Global_j_control_get ( );
        int8_t i_control = Werk ? Werk->i_control : GB_Global_i_control_get ( );

        // determine ideal 32/64 sizes for any matrix created by the caller
        bool p_prefer_32 = (p_control <= 32) ;
        bool j_prefer_32 = (j_control <= 32) ;
        bool i_prefer_32 = (i_control <= 32) ;

        // revise them accordering to the matrix content
        (*p_is_32) = GB_determine_p_is_32 (p_prefer_32, nvals) ;    // OK
        (*j_is_32) = GB_determine_j_is_32 (j_prefer_32, vdim) ;     // OK
        (*i_is_32) = GB_determine_i_is_32 (i_prefer_32, vlen) ;     // OK
    }
}

#endif

