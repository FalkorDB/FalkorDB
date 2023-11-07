//------------------------------------------------------------------------------
// GB_split_bitmap: split a bitmap matrix into an array of matrices
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// JIT: done.

#define GB_FREE_ALL         \
    GB_Matrix_free (&C) ;

#include "GB_split.h"
#include "GB_stringify.h"
#include "GB_apply.h"

GrB_Info GB_split_bitmap            // split a bitmap matrix
(
    GrB_Matrix *Tiles,              // 2D row-major array of size m-by-n
    const GrB_Index m,
    const GrB_Index n,
    const int64_t *restrict Tile_rows,  // size m+1
    const int64_t *restrict Tile_cols,  // size n+1
    const GrB_Matrix A,             // input matrix
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT (GB_IS_BITMAP (A)) ;
    GrB_Matrix C = NULL ;

    int sparsity_control = A->sparsity_control ;
    float hyper_switch = A->hyper_switch ;
    bool csc = A->is_csc ;
    GrB_Type atype = A->type ;
    int64_t avlen = A->vlen ;
//  int64_t avdim = A->vdim ;
    size_t asize = atype->size ;
    const int8_t *restrict Ab = A->b ;
    const bool A_iso = A->iso ;
//  int64_t anz = GB_nnz (A) ;

    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;

    int64_t nouter = csc ? n : m ;
    int64_t ninner = csc ? m : n ;

    const int64_t *Tile_vdim = csc ? Tile_cols : Tile_rows ;
    const int64_t *Tile_vlen = csc ? Tile_rows : Tile_cols ;

    //--------------------------------------------------------------------------
    // split A into tiles
    //--------------------------------------------------------------------------

    for (int64_t outer = 0 ; outer < nouter ; outer++)
    {

        const int64_t avstart = Tile_vdim [outer] ;
        const int64_t avend   = Tile_vdim [outer+1] ;

        for (int64_t inner = 0 ; inner < ninner ; inner++)
        {

            //------------------------------------------------------------------
            // allocate the tile C
            //------------------------------------------------------------------

            // The tile appears in vectors avstart:avend-1 of A, and indices
            // aistart:aiend-1.

            const int64_t aistart = Tile_vlen [inner] ;
            const int64_t aiend   = Tile_vlen [inner+1] ;
            const int64_t cvdim = avend - avstart ;
            const int64_t cvlen = aiend - aistart ;
            int64_t cnzmax = cvdim * cvlen ;

            C = NULL ;
            // set C->iso = A_iso       OK
            GB_OK (GB_new_bix (&C, // new header
                atype, cvlen, cvdim, GB_Ap_null, csc, GxB_BITMAP, false,
                hyper_switch, 0, cnzmax, true, A_iso)) ;
            int8_t *restrict Cb = C->b ;
            C->sparsity_control = sparsity_control ;
            C->hyper_switch = hyper_switch ;
            int C_nthreads = GB_nthreads (cnzmax, chunk, nthreads_max) ;

            //------------------------------------------------------------------
            // copy the tile from A into C
            //------------------------------------------------------------------

            info = GrB_NO_VALUE ;

            if (A_iso)
            { 

                //--------------------------------------------------------------
                // split an iso matrix A into an iso tile C
                //--------------------------------------------------------------

                // A is iso and so is C; copy the iso entry
                memcpy (C->x, A->x, asize) ;
                #define GB_ISO_SPLIT
                #define GB_COPY(pC,pA) ;
                #include "GB_split_bitmap_template.c"
                info = GrB_SUCCESS ;

            }
            else
            {

                //--------------------------------------------------------------
                // split a non-iso matrix A into an non-iso tile C
                //--------------------------------------------------------------

                #ifndef GBCOMPACT
                GB_IF_FACTORY_KERNELS_ENABLED
                { 
                    // no typecasting needed
                    switch (asize)
                    {
                        #undef  GB_COPY
                        #define GB_COPY(pC,pA) Cx [pC] = Ax [pA]

                        case GB_1BYTE : // uint8, int8, bool, or 1-byte user
                            #define GB_C_TYPE uint8_t
                            #define GB_A_TYPE uint8_t
                            #include "GB_split_bitmap_template.c"
                            info = GrB_SUCCESS ;
                            break ;

                        case GB_2BYTE : // uint16, int16, or 2-byte user
                            #define GB_C_TYPE uint16_t
                            #define GB_A_TYPE uint16_t
                            #include "GB_split_bitmap_template.c"
                            info = GrB_SUCCESS ;
                            break ;

                        case GB_4BYTE : // uint32, int32, float, or 4-byte user
                            #define GB_C_TYPE uint32_t
                            #define GB_A_TYPE uint32_t
                            #include "GB_split_bitmap_template.c"
                            info = GrB_SUCCESS ;
                            break ;

                        case GB_8BYTE : // uint64, int64, double, float complex,
                                        // or 8-byte user defined
                            #define GB_C_TYPE uint64_t
                            #define GB_A_TYPE uint64_t
                            #include "GB_split_bitmap_template.c"
                            info = GrB_SUCCESS ;
                            break ;

                        case GB_16BYTE : // double complex or 16-byte user
                            #define GB_C_TYPE GB_blob16
                            #define GB_A_TYPE GB_blob16
                            #include "GB_split_bitmap_template.c"
                            info = GrB_SUCCESS ;
                            break ;

                        default:;
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
                GB_Operator op = GB_unop_identity (atype, &op_header) ;
                ASSERT_OP_OK (op, "id op for split bitmap", GB0) ;
                info = GB_split_bitmap_jit (C, op, A, avstart, aistart,
                    C_nthreads) ;
            }

            //------------------------------------------------------------------
            // via the generic kernel
            //------------------------------------------------------------------

            if (info == GrB_NO_VALUE)
            { 
                // user-defined types
                #define GB_C_TYPE GB_void
                #define GB_A_TYPE GB_void
                #undef  GB_COPY
                #define GB_COPY(pC,pA)  \
                    memcpy (Cx + (pC)*asize, Ax +(pA)*asize, asize) ;
                #include "GB_split_bitmap_template.c"
                info = GrB_SUCCESS ;
            }

            if (info != GrB_SUCCESS)
            { 
                // out of memory, or other error
                GB_FREE_ALL ;
                return (info) ;
            }

            //------------------------------------------------------------------
            // conform the tile and save it in the Tiles array
            //------------------------------------------------------------------

            C->magic = GB_MAGIC ;
            ASSERT_MATRIX_OK (C, "C for GB_split", GB0) ;
            GB_OK (GB_conform (C, Werk)) ;
            if (csc)
            { 
                GB_TILE (Tiles, inner, outer) = C ;
            }
            else
            { 
                GB_TILE (Tiles, outer, inner) = C ;
            }
            C = NULL ;
        }
    }

    return (GrB_SUCCESS) ;
}

