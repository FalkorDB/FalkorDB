//------------------------------------------------------------------------------
// GB_convert.h: converting between sparsity structures
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_CONVERT_H
#define GB_CONVERT_H

// these parameters define the hyper_switch needed to ensure matrix stays
// either always hypersparse, or never hypersparse.
#define GB_ALWAYS_HYPER (1.0)
#define GB_NEVER_HYPER  (-1.0)

// determine the sparsity_control for a matrix
int GB_sparsity_control     // revised sparsity_control
(
    int sparsity_control,   // sparsity_control
    int64_t vdim            // A->vdim, or -1 to ignore this condition
) ;

// GB_sparsity: determine the current sparsity format of a matrix
static inline int GB_sparsity (GrB_Matrix A)
{
    if (A == NULL)
    {
        // if A is NULL, pretend it is sparse
        return (GxB_SPARSE) ;
    }
    else if (GB_IS_HYPERSPARSE (A))
    { 
        return (GxB_HYPERSPARSE) ;
    }
    else if (GB_IS_FULL (A))
    { 
        return (GxB_FULL) ;
    }
    else if (GB_IS_BITMAP (A))
    { 
        return (GxB_BITMAP) ;
    }
    else
    { 
        return (GxB_SPARSE) ;
    }
}

GrB_Info GB_convert_hyper_to_sparse // convert hypersparse to sparse
(
    GrB_Matrix A,           // matrix to convert from hypersparse to sparse
    bool do_burble          // if true, then burble is allowed
) ;

GrB_Info GB_convert_sparse_to_hyper // convert from sparse to hypersparse
(
    GrB_Matrix A,           // matrix to convert to hypersparse
    GB_Werk Werk
) ;

bool GB_convert_hyper_to_sparse_test    // test for hypersparse to sparse
(
    float hyper_switch,     // A->hyper_switch
    int64_t k,              // # of non-empty vectors of A (an estimate is OK)
    int64_t vdim            // A->vdim
) ;

bool GB_convert_sparse_to_hyper_test  // test sparse to hypersparse conversion
(
    float hyper_switch,     // A->hyper_switch
    int64_t k,              // # of non-empty vectors of A (an estimate is OK)
    int64_t vdim            // A->vdim
) ;

bool GB_convert_bitmap_to_sparse_test    // test for hyper/sparse to bitmap
(
    float bitmap_switch,    // A->bitmap_switch
    int64_t anz,            // # of entries in A = GB_nnz (A)
    int64_t vlen,           // A->vlen
    int64_t vdim            // A->vdim
) ;

bool GB_convert_s2b_test    // test for hyper/sparse to bitmap
(
    float bitmap_switch,    // A->bitmap_switch
    int64_t anz,            // # of entries in A = GB_nnz (A)
    int64_t vlen,           // A->vlen
    int64_t vdim            // A->vdim
) ;

GrB_Info GB_convert_full_to_sparse      // convert matrix from full to sparse
(
    GrB_Matrix A                // matrix to convert from full to sparse
) ;

GrB_Info GB_convert_full_to_bitmap      // convert matrix from full to bitmap
(
    GrB_Matrix A                // matrix to convert from full to bitmap
) ;

GrB_Info GB_convert_s2b    // convert sparse/hypersparse to bitmap
(
    GrB_Matrix A,               // matrix to convert from sparse to bitmap
    GB_Werk Werk
) ;

GrB_Info GB_convert_bitmap_to_sparse    // convert matrix from bitmap to sparse
(
    GrB_Matrix A,               // matrix to convert from bitmap to sparse
    GB_Werk Werk
) ;

GrB_Info GB_convert_bitmap_worker   // extract CSC/CSR or triplets from bitmap
(
    // outputs:
    int64_t *restrict Ap,        // vector pointers for CSC/CSR form
    int64_t *restrict Ai,        // indices for CSC/CSR or triplet form
    int64_t *restrict Aj,        // vector indices for triplet form
    GB_void *restrict Ax_new,    // values for CSC/CSR or triplet form
    int64_t *anvec_nonempty,        // # of non-empty vectors
    // inputs: not modified
    const GrB_Matrix A,             // matrix to extract; not modified
    GB_Werk Werk
) ;

GrB_Info GB_convert_any_to_bitmap   // convert to bitmap
(
    GrB_Matrix A,           // matrix to convert to bitmap
    GB_Werk Werk
) ;

void GB_convert_any_to_full     // convert any matrix to full
(
    GrB_Matrix A                // matrix to convert to full
) ;

GrB_Info GB_convert_any_to_hyper // convert to hypersparse
(
    GrB_Matrix A,           // matrix to convert to hypersparse
    GB_Werk Werk
) ;

GrB_Info GB_convert_any_to_sparse // convert to sparse
(
    GrB_Matrix A,           // matrix to convert to sparse
    GB_Werk Werk
) ;

GrB_Info GB_convert_to_nonfull      // ensure a matrix is not full
(
    GrB_Matrix A,
    GB_Werk Werk
) ;

/* ensure C is sparse or hypersparse */
#define GB_ENSURE_SPARSE(C)                                 \
{                                                           \
    if (GB_IS_BITMAP (C))                                   \
    {                                                       \
        /* convert C from bitmap to sparse */               \
        GB_OK (GB_convert_bitmap_to_sparse (C, Werk)) ;     \
    }                                                       \
    else if (GB_IS_FULL (C))                                \
    {                                                       \
        /* convert C from full to sparse */                 \
        GB_OK (GB_convert_full_to_sparse (C)) ;             \
    }                                                       \
}

//------------------------------------------------------------------------------
// GB_is_dense
//------------------------------------------------------------------------------

static inline bool GB_is_dense
(
    const GrB_Matrix A
)
{
    // check if A is competely dense:  all entries present.
    // zombies, pending tuples, and jumbled status are not considered.
    // A can have any sparsity structure: hyper, sparse, bitmap, or full.
    // It can be converted to full, if zombies/tuples/jumbled are discarded.
    if (A == NULL)
    { 
        return (false) ;
    }
    if (GB_IS_FULL (A))
    { 
        // A is full; the pattern is not present
        return (true) ;
    }
    // A is sparse, hyper, or bitmap: check if all entries present
    return (GB_nnz_full (A) == GB_nnz (A)) ;
}

//------------------------------------------------------------------------------
// GB_as_if_full
//------------------------------------------------------------------------------

static inline bool GB_as_if_full
(
    const GrB_Matrix A
)
{
    // check if A is competely dense:  all entries present.
    // zombies, pending tuples, and jumbled status are checked.
    // A can have any sparsity structure: hyper, sparse, bitmap, or full.
    // It can be converted to full.
    if (A == NULL)
    { 
        return (false) ;
    }
    if (GB_IS_FULL (A))
    { 
        // A is full; the pattern is not present
        return (true) ;
    }
    if (GB_ANY_PENDING_WORK (A))
    { 
        // A has pending work and so cannot be treated as if full.
        // The existence of the hyperhash is not considered in this test.
        return (false) ;
    }
    // A is sparse, hyper, or bitmap: check if all entries present
    return (GB_nnz_full (A) == GB_nnz (A)) ;
}

//------------------------------------------------------------------------------

GrB_Info GB_conform     // conform a matrix to its desired sparsity structure
(
    GrB_Matrix A,       // matrix to conform
    GB_Werk Werk
) ;

static inline const char *GB_sparsity_char (int sparsity)
{
    switch (sparsity)
    {
        case GxB_HYPERSPARSE: return ("H") ;
        case GxB_SPARSE:      return ("S") ;
        case GxB_BITMAP:      return ("B") ;
        case GxB_FULL:        return ("F") ;
        default: ASSERT (0) ; return ("?") ;
    }
}

static inline const char *GB_sparsity_char_matrix (GrB_Matrix A)
{
    bool A_as_if_full = GB_as_if_full (A) ;
    if (A == NULL)             return (".") ;
    if (GB_IS_HYPERSPARSE (A)) return (A_as_if_full ? "Hf" : "H") ;
    if (GB_IS_SPARSE (A))      return (A_as_if_full ? "Sf" : "S") ;
    if (GB_IS_BITMAP (A))      return (A_as_if_full ? "Bf" : "B") ;
    if (GB_IS_FULL (A))        return ("F") ;
    ASSERT (0) ;               return ("?") ;
}

GrB_Matrix GB_hyper_shallow         // return C
(
    GrB_Matrix C,                   // output matrix
    const GrB_Matrix A              // input matrix
) ;

#endif

