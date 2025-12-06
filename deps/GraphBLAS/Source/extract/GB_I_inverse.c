//------------------------------------------------------------------------------
// GB_I_inverse: invert an index list, by constructing R = inverse (I)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// I is a large list relative to the vector length, avlen, and it is not
// contiguous.  Construct the matrix R to hold the inverse of the I list for
// quick lookup.  If i = I [k1], i = I [k2], and i = I [k3], then row i of the
// R matrix holds entries in columns k1, k2, and k3.  R is iso-valued and held
// by row.  If R has enough entries, it is converted to sparse.  Otherwise, the
// hyper_hash R->Y is constructed to enable fast lookup of R(i,:).

#define GB_FREE_WORKSPACE                       \
{                                               \
    GB_FREE_MEMORY (&W, W_size) ;               \
}

#define GB_FREE_ALL                             \
{                                               \
    GB_FREE_WORKSPACE ;                         \
    GB_Matrix_free (&R) ;                       \
}

#include "extract/GB_subref.h"
#include "builder/GB_build.h"

GrB_Info GB_I_inverse           // invert the I list for C=A(I,:)
(
    const void *I,              // list of indices, duplicates OK
    const bool I_is_32,         // if true, I is 32-bit; else 64 bit
    int64_t nI,                 // length of I
    int64_t avlen,              // length of the vectors of A
    // outputs:
    GrB_Matrix *R_handle,       // R = inverse (I)
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    GrB_Info info = GrB_SUCCESS ;
    GrB_Matrix R = NULL ;
    GB_MDECL (W, , u) ; size_t W_size = 0 ;
    (*R_handle) = NULL ;
    GB_IDECL (I, const, u) ; GB_IPTR (I, I_is_32) ;

    //--------------------------------------------------------------------------
    // construct R matrix to hold the inverse of I
    //--------------------------------------------------------------------------

    int64_t rvdim = avlen ;
    int64_t rvlen = nI ;
    int64_t rnvals = nI ;
    bool Rp_is_32, Rj_is_32, Ri_is_32 ;
    GB_determine_pji_is_32 (&Rp_is_32, &Rj_is_32, &Ri_is_32,
        GxB_HYPERSPARSE, rnvals, rvlen, rvdim, Werk) ;

    bool W_is_32 = (nI < INT32_MAX) ;
    size_t wsize = (W_is_32) ? sizeof (uint32_t) : sizeof (uint64_t) ;
    W = GB_MALLOC_MEMORY (nI, wsize, &W_size) ;
    if (W == NULL)
    { 
        // out of memory
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    GB_IPTR (W, W_is_32) ;
    for (int64_t k = 0 ; k < nI ; k++)
    { 
        // W [k] = k
        GB_ISET (W, k, k) ;
    }

    // create R: rvdim-by-rvlen (avlen-by-nI), held by row, iso-valued
    GB_OK (GB_new (&R,  // new dynamic header, do not allocate content
        GrB_UINT64, rvlen, rvdim, GB_ph_null, false, GxB_HYPERSPARSE, -1, 0,
        Rp_is_32, Rj_is_32, Ri_is_32)) ;

    uint64_t S_input [1] ;
    S_input [0] = 1 ;

    void *no_I_work = NULL ; size_t I_work_size = 0 ;
    void *no_J_work = NULL ; size_t J_work_size = 0 ;
    GB_void *no_X_work = NULL ; size_t X_work_size = 0 ;

    GB_OK (GB_builder (
        // T
        R,                  // matrix to build, R of size rvdim-by-rvlen
        // ttype
        GrB_UINT64,         // type of R (iso-valued)
        // vlen
        rvlen,              // length of each vector of R (= nI)
        // vdim
        rvdim,              // number of vectors of R (= avlen)
        // is_csc
        false,              // R is CSR
        // I_work_handle and size
        &no_I_work, &I_work_size,            // I_work not used
        // J_work_handle and size
        &no_J_work, &J_work_size,            // J_work not used
        // X_work_handle and size
        &no_X_work, &X_work_size,            // X_work not used
        // known_sorted
        false,              // tuples might not be sorted
        // known_no_duplicates
        true,               // no duplicates are present (W is unique)
        // isjlen
        nI,                 // size of I and W arrays
        // is_matrix
        true,               // R is a matrix
        // I_input
        W,                  // column indices are W [0..nI-1] = (0:nI-1)
        // J_input
        I,                  // row indices are in I [0..nI-1]
        // S_input
        S_input,            // values of R (iso-valued)
        // S_iso
        true,               // R is iso-valued
        // nvals
        rnvals,             // # of tuples in R (= nI)
        // dup operator
        NULL,               // no dup operator
        // stype
        GrB_UINT64,         // type of S_input
        // do_burble
        true,               // allow burble
        Werk,
        W_is_32,            // true if W is 32-bit, false if 64
        I_is_32,            // true if I is 32-bit, false if 64
        Rp_is_32,           // true if R->p is built as 32-bit, false if 64
        Rj_is_32,           // true if R->h is built as 32-bit, false if 64
        Ri_is_32            // true if R->i is built as 32-bit, false if 64
        )) ;

    // R is hypersparse; convert to sparse if possible
    ASSERT (GB_IS_HYPERSPARSE (R)) ;
    // if needed, the # of duplicates in I is (nI - R->nvec)
    if (rvdim < 32 * R->nvec)
    { 
        // R is rvdim-by-rvlen in hypersparse CSR format.  Determine if it
        // should be held in a sparse format instead of hypersparse.  R takes
        // O(rnvals) memory as hypersparse and O(rnvals+rvdim) as sparse.
        // Switch R to sparse format if rvdim is small enough.
        GB_OK (GB_convert_hyper_to_sparse (R, true)) ;
    }
    else
    { 
        // Keep R as hypersparse, but build its R->Y hyper_hash matrix
        GB_OK (GB_hyper_hash_build (R, Werk)) ;
    }

    //--------------------------------------------------------------------------
    // check result
    //--------------------------------------------------------------------------

    #ifdef GB_DEBUG
    {
        // this test can take a very long time if A is hypersparse and
        // avlen is huge
        bool R_is_hyper = GB_IS_HYPERSPARSE (R) ;
        int64_t rnvec = R->nvec ;
        void *Rp = R->p ;
        void *Rh = R->h ;
        void *Ri = R->i ;
        GB_IDECL (Rp, const, u) ; GB_IPTR (Rp, Rp_is_32) ;
        GB_IDECL (Rh, const, u) ; GB_IPTR (Rh, Rj_is_32) ;
        GB_IDECL (Ri, const, u) ; GB_IPTR (Ri, Ri_is_32) ;
        GrB_Matrix R_Y = R->Y ;
        void *R_Yp = R_Y ? R_Y->p : NULL ;
        void *R_Yi = R_Y ? R_Y->i : NULL ;
        void *R_Yx = R_Y ? R_Y->x : NULL ;
        int64_t R_hash_bits = R_Y ? (R_Y->vdim - 1) : 0 ;
        for (int64_t i = 0 ; i < avlen ; i++)
        {
            // find R(i,:), which contains one column index inew for each
            // position in I where i occurs (i == I [inew])
            int64_t pR, pR_end ;
            if (R_is_hyper)
            {
                // R(i,:) is the kth vector in the hypersparse matrix R;
                // find k so that i = Rh [k] using the R->Y hyper_hash,
                // and set pR = Rp [k] and pR_end = Rp [k+1].
                GB_hyper_hash_lookup (Rp_is_32, Rj_is_32,
                    Rh, rnvec, Rp, R_Yp, R_Yi, R_Yx, R_hash_bits,
                    i, &pR, &pR_end) ;
            }
            else
            {
                // R(i,:) is the ith vector in the sparse matrix R
                pR = GB_IGET (Rp, i) ;          // pR = Rp [i]
                pR_end = GB_IGET (Rp, i+1) ;    // pR_end = Rp [i+1]
            }
            // for each entry in the row R(i,:)
            for (int64_t p = pR ; p < pR_end ; p++)
            {
                // get R(i,inew); this is the index i = I [inew]
                int64_t inew = GB_IGET (Ri, p) ;        // inew = Ri [p]
                ASSERT (inew >= 0 && inew < nI) ;
                ASSERT (i == GB_IGET (I, inew)) ;
            }
        }
    }
    #endif

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    GB_FREE_WORKSPACE ;
    ASSERT_MATRIX_OK (R, "R = I_inverse matrix", GB2) ;
    (*R_handle) = R ;
    return (GrB_SUCCESS) ;
}

