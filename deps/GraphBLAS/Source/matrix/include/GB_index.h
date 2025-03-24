//------------------------------------------------------------------------------
// GB_index.h: definitions for matrix indices and integers
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_INDEX_H
#define GB_INDEX_H

//------------------------------------------------------------------------------
// maximum matrix or vector dimension
//------------------------------------------------------------------------------

// GB_NMAX:   max dimension when A->i or A->h are 64-bit
// GB_NMAX32: max dimension when A->i or A->h are 32-bit
#define GB_NMAX   ((uint64_t) (1ULL << 60))
#define GB_NMAX32 ((uint64_t) (1ULL << 31))

//------------------------------------------------------------------------------
// GB_determine_p_is_32: returns a revised p_is_32, based on nvals_max
//------------------------------------------------------------------------------

static inline bool GB_determine_p_is_32
(
    bool p_is_32,       // if true, requesting 32-bit offsets
    int64_t nvals_max   // max # of entries in the matrix
)
{
    if (p_is_32 && nvals_max > UINT32_MAX)
    { 
        // A->p is requested too small; make it 64-bit
        p_is_32 = false ;
    }
    return (p_is_32) ;
}

//------------------------------------------------------------------------------
// GB_determine_j_is_32: returns a revised j_is_32, based on vdim dimension
//------------------------------------------------------------------------------

static inline bool GB_determine_j_is_32
(
    bool j_is_32,       // if true, requesting 32-bit indices
    int64_t vdim        // vector dimension of the matrix
)
{
    if (j_is_32 && vdim > GB_NMAX32)
    { 
        // A->h, A->Y is requested too small; make it 64-bit
        j_is_32 = false ;
    }
    return (j_is_32) ;
}

//------------------------------------------------------------------------------
// GB_determine_i_is_32: returns a revised i_is_32, based on vlen dimension
//------------------------------------------------------------------------------

static inline bool GB_determine_i_is_32
(
    bool i_is_32,       // if true, requesting 32-bit indices
    int64_t vlen        // vector length of the matrix
)
{
    if (i_is_32 && vlen > GB_NMAX32)
    { 
        // A->i is requested too small; make it 64-bit
        i_is_32 = false ;
    }
    return (i_is_32) ;
}

//------------------------------------------------------------------------------
// GB_pji_control: return effective p_control, j_control, or i_control
//------------------------------------------------------------------------------

static inline int8_t GB_pji_control
(
    int8_t matrix_pji_control,
    int8_t global_pji_control
)
{
    if (matrix_pji_control == 0)
    { 
        // default: matrix control defers to the global control
        return (global_pji_control) ;
    }
    else
    { 
        // use the matrix-specific cisontrol
        return (matrix_pji_control) ;
    }
}

//------------------------------------------------------------------------------
// GB_valid_[pji]_is_32: returns true if [pji] settings are OK
//------------------------------------------------------------------------------

// returns true if the pi settings are OK for this matrix.
// returns false if the matrix is too large for its pi settings

static inline bool GB_valid_p_is_32
(
    bool p_is_32,   // if true, A->p is 32-bit; else 64-bit
    int64_t nvals   // # of entries in the matrix
)
{ 
    // matrix is valid if A->p is 64 bit, or nvals is small enough
    return (!p_is_32 || nvals < UINT32_MAX) ;
}

static inline bool GB_valid_j_is_32
(
    bool j_is_32,   // if true, A->h and A->Y are 32-bit; else 64-bit
    int64_t vdim    // matrix dimension (# of vectors)
)
{ 
    // matrix is valid if A->h and A->Y are 64 bit, or vdim is small enough
    return (!j_is_32 || vdim <= GB_NMAX32) ;
}

static inline bool GB_valid_i_is_32
(
    bool i_is_32,   // if true, A->i is 32-bit; else 64-bit
    int64_t vlen    // matrix dimension (length of each vector)
)
{ 
    // matrix is valid if A->i is 64 bit, or vlen is small enough
    return (!i_is_32 || vlen <= GB_NMAX32) ;
}

static inline bool GB_valid_pji_is_32
(
    bool p_is_32,   // if true, A->p is 32-bit; else 64-bit
    bool j_is_32,   // if true, A->h and A->Y are 32-bit; else 64-bit
    bool i_is_32,   // if true, A->i is 32-bit; else 64-bit
    int64_t nvals,  // # of entries in the matrix
    int64_t vlen,   // matrix dimensions
    int64_t vdim
)
{ 
    // matrix is valid if A->p is 64 bit, or nvals is small enough, and
    // if A->i is 64 bit, or the dimensions are small enough.
    return (GB_valid_p_is_32 (p_is_32, nvals) &&
            GB_valid_j_is_32 (j_is_32, vdim) &&
            GB_valid_i_is_32 (i_is_32, vlen)) ;
}

//------------------------------------------------------------------------------
// GB_valid_matrix: check if a matrix is valid
//------------------------------------------------------------------------------

static inline GrB_Info GB_valid_matrix // returns GrB_SUCCESS, or error
(
    GrB_Matrix A                // matrix to validate
)
{

    // a NULL matrix is always valid so far (it may be an optional argument)
    if (A == NULL)
    { 
        return (GrB_SUCCESS) ;
    }

    // check the magic status
    if (A->magic != GB_MAGIC)
    { 
        if (A->magic == GB_MAGIC2)
        { 
            return (GrB_INVALID_OBJECT) ;
        }
        else
        { 
            return (GrB_UNINITIALIZED_OBJECT) ;
        }
    }

    // a full or bitmap matrix has no integers
    if (A->p == NULL && A->h == NULL && A->i == NULL && A->Y == NULL)
    { 
        return (GrB_SUCCESS) ;
    }

    // ensure that the matrix status is large enough for its content
    if (!GB_valid_pji_is_32 (A->p_is_32, A->j_is_32, A->i_is_32,
        A->nvals, A->vlen, A->vdim))
    { 
        return (GrB_INVALID_OBJECT) ;
    }

    return (GrB_SUCCESS) ;
}

#endif

