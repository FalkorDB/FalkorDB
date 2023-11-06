//------------------------------------------------------------------------------
// GB_concat_sparse: concatenate an array of matrices into a sparse matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// JIT: done.

#define GB_FREE_WORKSPACE                       \
    if (S != NULL)                              \
    {                                           \
        for (int64_t k = 0 ; k < m * n ; k++)   \
        {                                       \
            GB_Matrix_free (&(S [k])) ;         \
        }                                       \
    }                                           \
    GB_FREE_WORK (&S, S_size) ;                 \
    GB_FREE_WORK (&Work, Work_size) ;           \
    GB_WERK_POP (A_ek_slicing, int64_t) ;

#define GB_FREE_ALL         \
{                           \
    GB_FREE_WORKSPACE ;     \
    GB_phybix_free (C) ;    \
}

#include "GB_concat.h"
#include "GB_stringify.h"
#include "GB_apply.h"
#include "GB_unused.h"

GrB_Info GB_concat_sparse           // concatenate into a sparse matrix
(
    GrB_Matrix C,                   // input/output matrix for results
    const bool C_iso,               // if true, construct C as iso
    const GB_void *cscalar,         // iso value of C, if C is io 
    const int64_t cnz,              // # of entries in C
    const GrB_Matrix *Tiles,        // 2D row-major array of size m-by-n,
    const GrB_Index m,
    const GrB_Index n,
    const int64_t *restrict Tile_rows,  // size m+1
    const int64_t *restrict Tile_cols,  // size n+1
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // allocate C as a sparse matrix
    //--------------------------------------------------------------------------

    GrB_Info info ;
    GrB_Matrix A = NULL ;
    ASSERT_MATRIX_OK (C, "C input to concat sparse", GB0) ;
    GB_WERK_DECLARE (A_ek_slicing, int64_t) ;
    int64_t *Work = NULL ;
    size_t Work_size = 0 ;
    GrB_Matrix *S = NULL ;
    size_t S_size = 0 ;

    GrB_Type ctype = C->type ;
    int64_t cvlen = C->vlen ;
    int64_t cvdim = C->vdim ;
    bool csc = C->is_csc ;
    size_t csize = ctype->size ;
    GB_Type_code ccode = ctype->code ;

    float hyper_switch = C->hyper_switch ;
    float bitmap_switch = C->bitmap_switch ;
    int sparsity_control = C->sparsity_control ;
    GB_phybix_free (C) ;
    // set C->iso = C_iso   OK
    GB_OK (GB_new_bix (&C, // existing header
        ctype, cvlen, cvdim, GB_Ap_malloc, csc, GxB_SPARSE, false,
        hyper_switch, cvdim, cnz, true, C_iso)) ;
    C->bitmap_switch = bitmap_switch ;
    C->sparsity_control = sparsity_control ;
    int64_t *restrict Cp = C->p ;
    int64_t *restrict Ci = C->i ;

    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;

    if (C_iso)
    { 
        memcpy (C->x, cscalar, csize) ;
    }

    //--------------------------------------------------------------------------
    // allocate workspace
    //--------------------------------------------------------------------------

    int64_t nouter = csc ? n : m ;
    int64_t ninner = csc ? m : n ;
    Work = GB_CALLOC_WORK (ninner * cvdim, int64_t, &Work_size) ;
    S = GB_CALLOC_WORK (m * n, GrB_Matrix, &S_size) ;
    if (S == NULL || Work == NULL)
    { 
        // out of memory
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    //--------------------------------------------------------------------------
    // count entries in each vector of each tile
    //--------------------------------------------------------------------------

    for (int64_t outer = 0 ; outer < nouter ; outer++)
    {
        for (int64_t inner = 0 ; inner < ninner ; inner++)
        {

            //------------------------------------------------------------------
            // get the tile A; transpose and typecast, if needed
            //------------------------------------------------------------------

            A = csc ? GB_TILE (Tiles, inner, outer)
                    : GB_TILE (Tiles, outer, inner) ;
            GrB_Matrix T = NULL ;
            ASSERT_MATRIX_OK (A, "A tile for concat sparse", GB0) ;
            if (csc != A->is_csc)
            {
                // T = (ctype) A', not in-place, using a dynamic header
                GB_OK (GB_new (&T, // auto sparsity, new header
                    A->type, A->vdim, A->vlen, GB_Ap_null, csc,
                    GxB_AUTO_SPARSITY, -1, 1)) ;
                // save T in array S
                if (csc)
                { 
                    GB_TILE (S, inner, outer) = T ;
                }
                else
                { 
                    GB_TILE (S, outer, inner) = T ;
                }
                GB_OK (GB_transpose_cast (T, ctype, csc, A, false, Werk)) ;
                A = T ;
                GB_MATRIX_WAIT (A) ;
                ASSERT_MATRIX_OK (A, "T=A' for concat sparse", GB0) ;
            }
            ASSERT (C->is_csc == A->is_csc) ;
            ASSERT (!GB_ANY_PENDING_WORK (A)) ;

            //------------------------------------------------------------------
            // ensure the tile is not bitmap
            //------------------------------------------------------------------

            if (GB_IS_BITMAP (A))
            {
                if (T == NULL)
                {
                    // copy A into T
                    // set T->iso = A->iso  OK: no burble needed
                    GB_OK (GB_dup_worker (&T, A->iso, A, true, NULL)) ;
                    // save T in array S
                    if (csc)
                    { 
                        GB_TILE (S, inner, outer) = T ;
                    }
                    else
                    { 
                        GB_TILE (S, outer, inner) = T ;
                    }
                    ASSERT_MATRIX_OK (T, "T=dup(A) for concat sparse", GB0) ;
                }
                // convert T from bitmap to sparse
                GB_OK (GB_convert_bitmap_to_sparse (T, Werk)) ;
                ASSERT_MATRIX_OK (T, "T bitmap to sparse, concat sparse", GB0) ;
                A = T ;
            }

            ASSERT (!GB_IS_BITMAP (A)) ;

            //------------------------------------------------------------------
            // log the # of entries in each vector of the tile A
            //------------------------------------------------------------------

            const int64_t anvec = A->nvec ;
            const int64_t avlen = A->vlen ;
            int64_t cvstart = csc ?  Tile_cols [outer] : Tile_rows [outer] ;
            int64_t *restrict W = Work + inner * cvdim + cvstart ;
            int nth = GB_nthreads (anvec, chunk, nthreads_max) ;
            if (GB_IS_FULL (A))
            { 
                // A is full
                int64_t j ;
                #pragma omp parallel for num_threads(nth) schedule(static)
                for (j = 0 ; j < anvec ; j++)
                {
                    // W [j] = # of entries in A(:,j), which is just avlen
                    W [j] = avlen ;
                }
            }
            else
            { 
                // A is sparse or hyper
                int64_t k ;
                int64_t *restrict Ah = A->h ;
                int64_t *restrict Ap = A->p ;
                #pragma omp parallel for num_threads(nth) schedule(static)
                for (k = 0 ; k < anvec ; k++)
                {
                    // W [j] = # of entries in A(:,j), the kth column of A
                    int64_t j = GBH (Ah, k) ;
                    W [j] = Ap [k+1] - Ap [k] ; 
                }
            }
        }
    }

    //--------------------------------------------------------------------------
    // cumulative sum of entries in each tile
    //--------------------------------------------------------------------------

    int nth = GB_nthreads (ninner*cvdim, chunk, nthreads_max) ;
    int64_t k ;
    #pragma omp parallel for num_threads(nth) schedule(static)
    for (k = 0 ; k < cvdim ; k++)
    {
        int64_t s = 0 ;
        for (int64_t inner = 0 ; inner < ninner ; inner++)
        { 
            int64_t p = inner * cvdim + k ;
            int64_t c = Work [p] ;
            Work [p] = s ;
            s += c ;
        }
        // total number of entries in C(:,k)
        Cp [k] = s ;
    }

    GB_cumsum (Cp, cvdim, &(C->nvec_nonempty), nthreads_max, Werk) ;
    ASSERT (cnz == Cp [cvdim]) ;
    C->nvals = cnz ;

    #pragma omp parallel for num_threads(nth) schedule(static)
    for (k = 0 ; k < cvdim ; k++)
    {
        int64_t pC = Cp [k] ;
        for (int64_t inner = 0 ; inner < ninner ; inner++)
        { 
            int64_t p = inner * cvdim + k ;
            Work [p] += pC ;
        }
    }

    //--------------------------------------------------------------------------
    // concatenate all matrices into C
    //--------------------------------------------------------------------------

    for (int64_t outer = 0 ; outer < nouter ; outer++)
    {
        for (int64_t inner = 0 ; inner < ninner ; inner++)
        {

            //------------------------------------------------------------------
            // get the tile A, either the temporary matrix T or the original A
            //------------------------------------------------------------------

            A = csc ? GB_TILE (S, inner, outer)
                    : GB_TILE (S, outer, inner) ;
            if (A == NULL)
            { 
                A = csc ? GB_TILE (Tiles, inner, outer)
                        : GB_TILE (Tiles, outer, inner) ;
            }
            ASSERT_MATRIX_OK (A, "A tile again, concat sparse", GB0) ;

            ASSERT (!GB_IS_BITMAP (A)) ;
            ASSERT (C->is_csc == A->is_csc) ;
            ASSERT (!GB_ANY_PENDING_WORK (A)) ;
            GB_Type_code acode = A->type->code ;

            //------------------------------------------------------------------
            // determine where to place the tile in C
            //------------------------------------------------------------------

            // The tile A appears in vectors cvstart:cvend-1 of C, and indices
            // cistart:ciend-1.

            int64_t cvstart, cvend, cistart, ciend ;
            if (csc)
            { 
                // C and A are held by column
                // Tiles is row-major and accessed in column order
                cvstart = Tile_cols [outer] ;
                cvend   = Tile_cols [outer+1] ;
                cistart = Tile_rows [inner] ;
                ciend   = Tile_rows [inner+1] ;
            }
            else
            { 
                // C and A are held by row
                // Tiles is row-major and accessed in row order
                cvstart = Tile_rows [outer] ;
                cvend   = Tile_rows [outer+1] ;
                cistart = Tile_cols [inner] ;
                ciend   = Tile_cols [inner+1] ;
            }

            // get the workspace pointer array W for this tile
            int64_t *restrict W = Work + inner * cvdim + cvstart ;

            //------------------------------------------------------------------
            // slice the tile
            //------------------------------------------------------------------

            int64_t avdim = cvend - cvstart ;
            int64_t avlen = ciend - cistart ;
            ASSERT (avdim == A->vdim) ;
            ASSERT (avlen == A->vlen) ;
            int A_nthreads, A_ntasks ;
            const int64_t *restrict Ap = A->p ;
            const int64_t *restrict Ah = A->h ;
            const int64_t *restrict Ai = A->i ;
            const bool A_iso = A->iso ;
            GB_SLICE_MATRIX (A, 1) ;

            //------------------------------------------------------------------
            // copy the tile A into C
            //------------------------------------------------------------------

            info = GrB_NO_VALUE ;

            if (C_iso)
            { 

                //--------------------------------------------------------------
                // C and A are iso
                //--------------------------------------------------------------

                #define GB_ISO_CONCAT
                #define GB_COPY(pC,pA,A_iso) ;
                #include "GB_concat_sparse_template.c"
                info = GrB_SUCCESS ;

            }
            else
            {

                //--------------------------------------------------------------
                // via the factory kernel (inline; not in FactoryKernels folder)
                //--------------------------------------------------------------

                #ifndef GBCOMPACT
                GB_IF_FACTORY_KERNELS_ENABLED
                { 
                    if (ccode == acode)
                    {
                        // no typecasting needed
                        switch (csize)
                        {
                            #undef  GB_COPY
                            #define GB_COPY(pC,pA,A_iso)    \
                                Cx [pC] = GBX (Ax, pA, A_iso) ;

                            case GB_1BYTE : // uint8, int8, bool, or 1-byte user
                                #define GB_C_TYPE uint8_t
                                #define GB_A_TYPE uint8_t
                                #include "GB_concat_sparse_template.c"
                                info = GrB_SUCCESS ;
                                break ;

                            case GB_2BYTE : // uint16, int16, or 2-byte user
                                #define GB_C_TYPE uint16_t
                                #define GB_A_TYPE uint16_t
                                #include "GB_concat_sparse_template.c"
                                info = GrB_SUCCESS ;
                                break ;

                            case GB_4BYTE : // uint32, int32, float, or 4-byte
                                #define GB_C_TYPE uint32_t
                                #define GB_A_TYPE uint32_t
                                #include "GB_concat_sparse_template.c"
                                info = GrB_SUCCESS ;
                                break ;

                            case GB_8BYTE : // uint64, int64, double, float
                                            // complex, or 8-byte user defined
                                #define GB_C_TYPE uint64_t
                                #define GB_A_TYPE uint64_t
                                #include "GB_concat_sparse_template.c"
                                info = GrB_SUCCESS ;
                                break ;

                            case GB_16BYTE : // double complex or 16-byte user
                                #define GB_C_TYPE GB_blob16
                                #define GB_A_TYPE GB_blob16
                                #include "GB_concat_sparse_template.c"
                                info = GrB_SUCCESS ;
                                break ;

                            default:;
                        }
                    }
                }
                #endif
            }

            //------------------------------------------------------------------
            // via the JIT or PreJIT kernel
            //------------------------------------------------------------------

            if (info == GrB_NO_VALUE)
            { 
                struct GB_UnaryOp_opaque op_header ;
                GB_Operator op = GB_unop_identity (ctype, &op_header) ;
                ASSERT_OP_OK (op, "identity op for concat sparse", GB0) ;
                info = GB_concat_sparse_jit (C, cistart, op, A, W,
                    A_ek_slicing, A_ntasks, A_nthreads) ;
            }

            //------------------------------------------------------------------
            // via the generic kernel
            //------------------------------------------------------------------

            if (info == GrB_NO_VALUE)
            { 
                // with typecasting or user-defined types
                GB_cast_function cast_A_to_C = GB_cast_factory (ccode, acode) ;
                size_t asize = A->type->size ;
                #define GB_C_TYPE GB_void
                #define GB_A_TYPE GB_void
                #undef  GB_COPY
                #define GB_COPY(pC,pA,A_iso)                    \
                    cast_A_to_C (Cx + (pC)*csize,               \
                        Ax + (A_iso ? 0:(pA)*asize), asize) ;
                #include "GB_concat_sparse_template.c"
                info = GrB_SUCCESS ;
            }
    
            GB_WERK_POP (A_ek_slicing, int64_t) ;

            if (info != GrB_SUCCESS)
            { 
                // out of memory, or other error
                GB_FREE_ALL ;
                return (info) ;
            }
        }
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    GB_FREE_WORKSPACE ;
    C->magic = GB_MAGIC ;
    ASSERT_MATRIX_OK (C, "C from concat sparse", GB0) ;
    return (GrB_SUCCESS) ;
}

