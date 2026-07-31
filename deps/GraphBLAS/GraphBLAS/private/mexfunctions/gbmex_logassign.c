//------------------------------------------------------------------------------
// gbmex_logassign: logical assignment: C(M) = A
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// gbmex_logassign computes the built-in logical indexing expression C(M) = A.
// The matrices C and M must be the same size.  M is normally logical but it
// can be of any type in this mexFunction.  M should not have any explicit
// zeros.  A is a sparse vector of size nnz(M)-by-1.  Scalar expansion is not
// handled.  Use GrB.subassign (C, M, scalar) for that case.

// This function accesses opaque content and GB_methods inside GraphBLAS.

// Usage:

//      C = gbmex_logassign (ghb, C, M, A)
//      gbmex_logassign (1, C, M, A)

//  This function is the C equivalent of the following m-function:

/*

    function C = gbmex_logassign (C, M_input, A)
    % Computing the built-in logical indexing expression C(M) = A in GraphBLAS.
    % A is a sparse vector of size nnz(M)-by-1 (scalar expansion is not
    % handled). M is normally a sparse logical matrix, either GraphBLAS or
    % built-in, but it can be of any type.  C and M have the same size.

    % make sure all matrices are stored by column
    save = GrB.format ;
    GrB.format ('by col') ;
    M = GrB (m, n, 'logical') ;
    M = GrB.select (M, '2nd', 'nonzero', M_input) ;
    if (isequal (GrB.format (A), 'by row'))
        A = GrB (A) ;
    end

    [m n] = size (C) ;
    mnz = nnz (M) ;         % A must be mnz-by-1
    if (~isequal (size (A), [mnz 1]))
        error ('GrB:error', 'A must be nnz(M)-by-1')
    end

    [ai,  ~, ax] = GrB.extracttuples (A) ;
    [mi, mj,  ~] = GrB.extracttuples (M) ;

    % construct a subset of the entries of the mask M corresponding to the
    % entries in A
    si = mi (ai) ;
    sj = mj (ai) ;
    S = GrB.build (si, sj, ax, m, n) ;

    GrB.format (save) ;

    % C<M> = S
    C = GrB.subassign (C, M, S) ;

*/

// This C mexFunction is faster than the above m-function, since it avoids the
// use of GrB.extracttuples.  Instead, it accesses the internal structure of the
// GrB_Matrix objects.  The m-file above is useful for understanding that this
// C mexFunction does.

// C is always returned as a GrB matrix.

#include "gb_interface.h"

#include "gbmx_interface.h"

#undef  FREE_WORK
#define FREE_WORK                       \
    gb_free ((void **) (&Si), arena) ;  \
    gb_free ((void **) (&Sj), arena) ;  \
    gb_free ((void **) (&Mj), arena) ;  \
    GrB_Matrix_free (&S) ;              \
    GrB_Matrix_free (&M) ;              \
    GrB_Matrix_free (&M_to_free) ;      \
    GrB_Matrix_free (&A_to_free) ;      \
    GrB_Matrix_free (&A_copy) ;         \
    GrB_Matrix_free (&A_copy2) ;

#undef  FREE_ALL
#define FREE_ALL                        \
    FREE_WORK ;                         \
    GrB_Matrix_free (&C) ;

#define USAGE "usage: C = gbmex_logassign (ghb, C, M, A)"
#define ERR "A must be a vector of length nnz(M) for logical indexing, C(M)=A"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    //--------------------------------------------------------------------------
    // check inputs and construct outputs
    //--------------------------------------------------------------------------

    GrB_Matrix *C_opaque = NULL, C = NULL, M = NULL, M_to_free = NULL,
        A = NULL, M_input = NULL, A_to_free = NULL, A_copy = NULL,
        A_copy2 = NULL, S = NULL ;
    uint64_t *Si = NULL, *Sj = NULL, *Mj = NULL ;

    GBMX_USAGE (nargin == 4 && nargout <= 1, USAGE) ;
    bool ghb = (bool) mxGetScalar (pargin [0]) ;
    int arena = ghb ? GrB_DEFAULT : MXARENA ;

    bool inplace = ghb && (nargout == 0) ;
    if (!inplace)
    { 
        if (ghb) pargout [0] = gbmx_export_ghb_mxstruct (&C_opaque) ;
    }
    else
    { 
        /* for tracking test coverage */ ;
    }

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    struct gb_matrix_struct Matrix [3] ;
    gbmx_get_matrix (&(Matrix [0]), pargin [1]) ;
    gbmx_get_matrix (&(Matrix [1]), pargin [2]) ;
    gbmx_get_matrix (&(Matrix [2]), pargin [3]) ;

    ////////////////////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    // get a deep copy of C, of any sparsity structure
    //--------------------------------------------------------------------------

    OK (gb_get_deep (&C, inplace, &(Matrix [0]), arena, err)) ;
    uint64_t nrows, ncols ;
    OK (GrB_Matrix_nrows (&nrows, C)) ;
    OK (GrB_Matrix_ncols (&ncols, C)) ;

    //--------------------------------------------------------------------------
    // get M
    //--------------------------------------------------------------------------

    // make M boolean, sparse/hyper, stored by column, and drop explicit zeros
    OK (gb_get_matrix (&M_input, &M_to_free, &(Matrix [1]), arena, err)) ;

    OK (gb_new (&M, GrB_BOOL, nrows, ncols, GxB_BY_COL,
        GxB_SPARSE + GxB_HYPERSPARSE, arena, err)) ;
    OK1 (M, GrB_Matrix_select_BOOL (M, NULL, NULL, GrB_VALUENE_BOOL, M_input,
        0, NULL)) ;

    GrB_Matrix_free (&M_to_free) ;
    uint64_t mnz ;
    OK (GrB_Matrix_nvals (&mnz, M)) ;

    //--------------------------------------------------------------------------
    // get A
    //--------------------------------------------------------------------------

    OK (gb_get_matrix (&A, &A_to_free, &(Matrix [2]), arena, err)) ;

    GrB_Type atype ;
    uint64_t anrows, ancols, anz ;
    int fmt ;
    int A_sparsity ;
    OK (GrB_Matrix_nrows (&anrows, A)) ;
    OK (GrB_Matrix_ncols (&ancols, A)) ;
    OK (GxB_Matrix_type (&atype, A)) ;
    OK (GrB_Matrix_nvals (&anz, A)) ;
    OK (GrB_Matrix_get_INT32 (A, &fmt, GxB_FORMAT)) ;
    OK (GrB_Matrix_get_INT32 (A, &A_sparsity, GxB_SPARSITY_STATUS)) ;

    // make sure A is not bitmap; it can be sparse, hypersparse, or full
    if (A_sparsity == GxB_BITMAP)
    { 
        OK (GxB_Matrix_dup_arena (&A_copy2, A, arena, arena)) ;
        OK1 (A_copy2, GrB_Matrix_set_INT32 (A_copy2,
            GxB_SPARSE + GxB_HYPERSPARSE + GxB_FULL, GxB_SPARSITY_CONTROL)) ;
        A = A_copy2 ;
    }

    // make sure A is a vector of the right size
    if (mnz == 0)
    { 
        // M is empty, so A must have no entries.  The dimensions and format of
        // A are not relevant, since the content of A will not be accessed.
        CHECK_ERROR (anz != 0, ERR) ;
    }
    else if (anrows == 1)
    { 
        // A is 1-by-ancols; ensure it is has length nnz(M), and held by row,
        // or transpose to ancols-by-1 and held by column.
        CHECK_ERROR (ancols != mnz, ERR) ;
        if (fmt == GxB_BY_COL)
        { 
            // A is 1-by-ancols and held by column: transpose it
            OK (gb_new (&A_copy, atype, mnz, 1, GxB_BY_COL,
                GxB_SPARSE + GxB_HYPERSPARSE + GxB_FULL, arena, err)) ;
            OK1 (A_copy, GrB_transpose (A_copy, NULL, NULL, A, NULL)) ;
            OK1 (A_copy, GrB_Matrix_wait (A_copy, GrB_MATERIALIZE)) ;
            A = A_copy ;
        }
    }
    else if (ancols == 1)
    { 
        // A is anrows-by-1; ensure it is has length nnz(M), and held by col
        // or transpose to 1-by-anrows and held by row.
        CHECK_ERROR (anrows != mnz, ERR) ;
        if (fmt == GxB_BY_ROW)
        { 
            // A is anrows-by-1 and held by row: transpose it
            OK (gb_new (&A_copy, atype, 1, mnz, GxB_BY_ROW,
                GxB_SPARSE + GxB_HYPERSPARSE + GxB_FULL, arena, err)) ;
            OK1 (A_copy, GrB_transpose (A_copy, NULL, NULL, A, NULL)) ;
            OK1 (A_copy, GrB_Matrix_wait (A_copy, GrB_MATERIALIZE)) ;
            A = A_copy ;
        }
    }
    else
    {
        ERROR (ERR, GrB_DIMENSION_MISMATCH) ;
    }

    //--------------------------------------------------------------------------
    // extract the values and pattern of A; handle iso case
    //--------------------------------------------------------------------------

    void *Ax = A->x ;
    char nil [16] = "iso logassign  " ;
    if (Ax == NULL) Ax = &nil ;

    //--------------------------------------------------------------------------
    // extract the pattern of M
    //--------------------------------------------------------------------------

    // FUTURE: use GxB_Matrix_extractTuples_Vector so Mj can be 32-bit

    Mj = gb_malloc (mnz * sizeof (uint64_t), arena) ;
    if (Mj == NULL) ERROR ("out of memory", GrB_OUT_OF_MEMORY) ;

    OK (GrB_Matrix_extractTuples_BOOL (NULL, Mj, NULL, &mnz, M)) ;

    //--------------------------------------------------------------------------
    // construct a subset of the pattern of M corresponding to the entries of A
    //--------------------------------------------------------------------------

    // FUTURE: allow Si and Sj to be 32-bit

    Si = gb_malloc (anz * sizeof (uint64_t), arena) ;
    Sj = gb_malloc (anz * sizeof (uint64_t), arena) ;
    if (Si == NULL || Sj == NULL) ERROR ("out of memory", GrB_OUT_OF_MEMORY) ;

    OK (GB_helper5 (Si, Sj, M->i, M->i_is_32, Mj, M->vlen, A->i, A->i_is_32,
        A->vlen, anz)) ;
    OK (gb_new (&S, atype, nrows, ncols, GxB_BY_COL, 0, arena, err)) ;

    if (A->iso)
    { 
        // build S as an iso matrix
        GrB_Scalar s = NULL ;
        OK (GxB_Scalar_new_arena (&s, atype, arena, arena)) ;
        if (atype == GrB_BOOL)
        { 
            OK (GrB_Scalar_setElement_BOOL (s, (* ((bool *) Ax)))) ;
        }
        else if (atype == GrB_INT8)
        { 
            OK (GrB_Scalar_setElement_INT8 (s, (* ((int8_t *) Ax)))) ;
        }
        else if (atype == GrB_INT16)
        { 
            OK (GrB_Scalar_setElement_INT16 (s, (* ((int16_t *) Ax)))) ;
        }
        else if (atype == GrB_INT32)
        { 
            OK (GrB_Scalar_setElement_INT32 (s, (* ((int32_t *) Ax)))) ;
        }
        else if (atype == GrB_INT64)
        { 
            OK (GrB_Scalar_setElement_INT64 (s, (* ((int64_t *) Ax)))) ;
        }
        else if (atype == GrB_UINT8)
        { 
            OK (GrB_Scalar_setElement_UINT8 (s, (* ((uint8_t *) Ax)))) ;
        }
        else if (atype == GrB_UINT16)
        { 
            OK (GrB_Scalar_setElement_UINT16 (s, (* ((uint16_t *) Ax)))) ;
        }
        else if (atype == GrB_UINT32)
        { 
            OK (GrB_Scalar_setElement_UINT32 (s, (* ((uint32_t *) Ax)))) ;
        }
        else if (atype == GrB_UINT64)
        { 
            OK (GrB_Scalar_setElement_UINT64 (s, (* ((uint64_t *) Ax)))) ;
        }
        else if (atype == GrB_FP32)
        { 
            OK (GrB_Scalar_setElement_FP32 (s, (* ((float *) Ax)))) ;
        }
        else if (atype == GrB_FP64)
        { 
            OK (GrB_Scalar_setElement_FP64 (s, (* ((double *) Ax)))) ;
        }
        else if (atype == GxB_FC32)
        { 
            OK (GxB_Scalar_setElement_FC32 (s, (* ((GxB_FC32_t *) Ax)))) ;
        }
        else if (atype == GxB_FC64)
        { 
            OK (GxB_Scalar_setElement_FC64 (s, (* ((GxB_FC64_t *) Ax)))) ;
        }
        else
        {
            ERROR ("unsupported type", GrB_DOMAIN_MISMATCH) ;
        }
        OK1 (S, GxB_Matrix_build_Scalar (S, Si, Sj, s, anz)) ;
        GrB_Scalar_free (&s) ;
    }
    else if (atype == GrB_BOOL)
    { 
        OK1 (S, GrB_Matrix_build_BOOL (S, Si, Sj, Ax, anz, GrB_LOR)) ;
    }
    else if (atype == GrB_INT8)
    { 
        OK1 (S, GrB_Matrix_build_INT8 (S, Si, Sj, Ax, anz, GrB_PLUS_INT8)) ;
    }
    else if (atype == GrB_INT16)
    { 
        OK1 (S, GrB_Matrix_build_INT16 (S, Si, Sj, Ax, anz, GrB_PLUS_INT16)) ;
    }
    else if (atype == GrB_INT32)
    { 
        OK1 (S, GrB_Matrix_build_INT32 (S, Si, Sj, Ax, anz, GrB_PLUS_INT32)) ;
    }
    else if (atype == GrB_INT64)
    { 
        OK1 (S, GrB_Matrix_build_INT64 (S, Si, Sj, Ax, anz, GrB_PLUS_INT64)) ;
    }
    else if (atype == GrB_UINT8)
    { 
        OK1 (S, GrB_Matrix_build_UINT8 (S, Si, Sj, Ax, anz, GrB_PLUS_UINT8)) ;
    }
    else if (atype == GrB_UINT16)
    { 
        OK1 (S, GrB_Matrix_build_UINT16 (S, Si, Sj, Ax, anz, GrB_PLUS_UINT16)) ;
    }
    else if (atype == GrB_UINT32)
    { 
        OK1 (S, GrB_Matrix_build_UINT32 (S, Si, Sj, Ax, anz, GrB_PLUS_UINT32)) ;
    }
    else if (atype == GrB_UINT64)
    { 
        OK1 (S, GrB_Matrix_build_UINT64 (S, Si, Sj, Ax, anz, GrB_PLUS_UINT64)) ;
    }
    else if (atype == GrB_FP32)
    { 
        OK1 (S, GrB_Matrix_build_FP32 (S, Si, Sj, Ax, anz, GrB_PLUS_FP32)) ;
    }
    else if (atype == GrB_FP64)
    { 
        OK1 (S, GrB_Matrix_build_FP64 (S, Si, Sj, Ax, anz, GrB_PLUS_FP64)) ;
    }
    else if (atype == GxB_FC32)
    { 
        OK1 (S, GxB_Matrix_build_FC32 (S, Si, Sj, Ax, anz, GxB_PLUS_FC32)) ;
    }
    else if (atype == GxB_FC64)
    { 
        OK1 (S, GxB_Matrix_build_FC64 (S, Si, Sj, Ax, anz, GxB_PLUS_FC64)) ;
    }
    else
    {
        ERROR ("unsupported type", GrB_DOMAIN_MISMATCH) ;
    }

    GrB_Matrix_free (&A_copy) ;
    GrB_Matrix_free (&A_copy2) ;

    //--------------------------------------------------------------------------
    // C<M> = S
    //--------------------------------------------------------------------------

    OK1 (C, GxB_Matrix_subassign (C, M, NULL,
        S, GrB_ALL, nrows, GrB_ALL, ncols, NULL)) ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    FREE_WORK ;

    if (!inplace)
    { 
        OK (gb_export (C_opaque, &C, KIND_GRB, ghb, err)) ;
    }
    ////////////////////////////////////////////////////////////////////////////
    if (!ghb)
    { 
        pargout [0] = gbmx_export_grb_mxstruct (&C) ;
    }

    gb_wrapup ( ) ;
}

