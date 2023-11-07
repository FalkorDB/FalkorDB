//------------------------------------------------------------------------------
// GB_concat_full: concatenate an array of matrices into a full matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// JIT: done.

#define GB_FREE_WORKSPACE   \
    GB_Matrix_free (&T) ;

#define GB_FREE_ALL         \
    GB_FREE_WORKSPACE ;     \
    GB_phybix_free (C) ;

#include "GB_concat.h"
#include "GB_stringify.h"
#include "GB_apply.h"

GrB_Info GB_concat_full             // concatenate into a full matrix
(
    GrB_Matrix C,                   // input/output matrix for results
    const bool C_iso,               // if true, construct C as iso
    const GB_void *cscalar,         // iso value of C, if C is io 
    const GrB_Matrix *Tiles,        // 2D row-major array of size m-by-n,
    const GrB_Index m,
    const GrB_Index n,
    const int64_t *restrict Tile_rows,  // size m+1
    const int64_t *restrict Tile_cols,  // size n+1
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // allocate C as a full matrix
    //--------------------------------------------------------------------------

    GrB_Info info ;
    GrB_Matrix A = NULL ;
    struct GB_Matrix_opaque T_header ;
    GrB_Matrix T = NULL ;

    GrB_Type ctype = C->type ;
    int64_t cvlen = C->vlen ;
    int64_t cvdim = C->vdim ;
    bool csc = C->is_csc ;
    size_t csize = ctype->size ;
    GB_Type_code ccode = ctype->code ;
    if (!GB_IS_FULL (C))
    { 
        // set C->iso = C_iso   OK
        GB_phybix_free (C) ;
        GB_OK (GB_bix_alloc (C, GB_nnz_full (C), GxB_FULL, false, true, C_iso));
        C->plen = -1 ;
        C->nvec = cvdim ;
        C->nvec_nonempty = (cvlen > 0) ? cvdim : 0 ;
    }
    ASSERT (GB_IS_FULL (C)) ;
    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;

    int64_t nouter = csc ? n : m ;
    int64_t ninner = csc ? m : n ;

    if (C_iso)
    { 
        // copy in the scalar as the iso value; no more work to do
        memcpy (C->x, cscalar, csize) ;
        C->magic = GB_MAGIC ;
        ASSERT_MATRIX_OK (C, "C output for concat iso full", GB0) ;
        return (GrB_SUCCESS) ;
    }

    //--------------------------------------------------------------------------
    // concatenate all matrices into C
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
            ASSERT (GB_IS_FULL (A)) ;
            if (csc != A->is_csc)
            { 
                // T = (ctype) A', not in-place
                GB_CLEAR_STATIC_HEADER (T, &T_header) ;
                GB_OK (GB_transpose_cast (T, ctype, csc, A, false, Werk)) ;
                A = T ;
                GB_MATRIX_WAIT (A) ;
            }
            ASSERT (C->is_csc == A->is_csc) ;
            ASSERT (GB_IS_FULL (A)) ;
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

            int64_t avdim = cvend - cvstart ;
            int64_t avlen = ciend - cistart ;
            ASSERT (avdim == A->vdim) ;
            ASSERT (avlen == A->vlen) ;
            int64_t anz = avdim * avlen ;
            int A_nthreads = GB_nthreads (anz, chunk, nthreads_max) ;

            //------------------------------------------------------------------
            // copy the tile A into C
            //------------------------------------------------------------------

            info = GrB_NO_VALUE ;

            //------------------------------------------------------------------
            // via the factory kernel (inline; not in FactoryKernels folder)
            //------------------------------------------------------------------

            #ifndef GBCOMPACT
            GB_IF_FACTORY_KERNELS_ENABLED
            { 
                if (ccode == acode)
                {
                    // no typecasting needed
                    switch (csize)
                    {
                        #define GB_COPY(pC,pA,A_iso)                        \
                            Cx [pC] = GBX (Ax, pA, A_iso) ;

                        case GB_1BYTE : // uint8, int8, bool, or 1-byte user
                            #define GB_C_TYPE uint8_t
                            #define GB_A_TYPE uint8_t
                            #include "GB_concat_full_template.c"
                            info = GrB_SUCCESS ;
                            break ;

                        case GB_2BYTE : // uint16, int16, or 2-byte user
                            #define GB_C_TYPE uint16_t
                            #define GB_A_TYPE uint16_t
                            #include "GB_concat_full_template.c"
                            info = GrB_SUCCESS ;
                            break ;

                        case GB_4BYTE : // uint32, int32, float, or 4-byte user
                            #define GB_C_TYPE uint32_t
                            #define GB_A_TYPE uint32_t
                            #include "GB_concat_full_template.c"
                            info = GrB_SUCCESS ;
                            break ;

                        case GB_8BYTE : // uint64, int64, double, float complex,
                                        // or 8-byte user defined
                            #define GB_C_TYPE uint64_t
                            #define GB_A_TYPE uint64_t
                            #include "GB_concat_full_template.c"
                            info = GrB_SUCCESS ;
                            break ;

                        case GB_16BYTE : // double complex or 16-byte user
                            #define GB_C_TYPE GB_blob16
                            #define GB_A_TYPE GB_blob16
                            #include "GB_concat_full_template.c"
                            info = GrB_SUCCESS ;
                            break ;

                        default:;
                    }
                }
            }
            #endif

            //------------------------------------------------------------------
            // via the JIT or PreJIT kernel
            //------------------------------------------------------------------

            if (info == GrB_NO_VALUE)
            { 
                struct GB_UnaryOp_opaque op_header ;
                GB_Operator op = GB_unop_identity (ctype, &op_header) ;
                ASSERT_OP_OK (op, "identity op for concat full", GB0) ;
                info = GB_concat_full_jit (C, cistart, cvstart, op, A,
                    A_nthreads) ;
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
                #include "GB_concat_full_template.c"
                info = GrB_SUCCESS ;
            }

            GB_FREE_WORKSPACE ;

            if (info != GrB_SUCCESS)
            { 
                // out of memory, or other error
                GB_FREE_ALL ;
                return (info) ;
            }
        }
    }

    C->magic = GB_MAGIC ;
    ASSERT_MATRIX_OK (C, "C output for concat full", GB0) ;
    return (GrB_SUCCESS) ;
}

