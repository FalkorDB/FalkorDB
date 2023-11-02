//------------------------------------------------------------------------------
// GB_split_sparse: split a sparse/hypersparse matrix into tiles 
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// JIT: done.

#define GB_FREE_WORKSPACE                   \
    GB_WERK_POP (C_ek_slicing, int64_t) ;   \
    GB_FREE_WORK (&Wp, Wp_size) ;

#define GB_FREE_ALL                         \
    GB_FREE_WORKSPACE ;                     \
    GB_Matrix_free (&C) ;

#include "GB_split.h"
#include "GB_stringify.h"
#include "GB_apply.h"

GrB_Info GB_split_sparse            // split a sparse matrix
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
    int A_sparsity = GB_sparsity (A) ;
    bool A_is_hyper = (A_sparsity == GxB_HYPERSPARSE) ;
    ASSERT (A_is_hyper || A_sparsity == GxB_SPARSE) ;
    GrB_Matrix C = NULL ;
    GB_WERK_DECLARE (C_ek_slicing, int64_t) ;
    ASSERT_MATRIX_OK (A, "A sparse for split", GB0) ;

    int sparsity_control = A->sparsity_control ;
    float hyper_switch = A->hyper_switch ;
    bool csc = A->is_csc ;
    GrB_Type atype = A->type ;
//  int64_t avlen = A->vlen ;
//  int64_t avdim = A->vdim ;
    size_t asize = atype->size ;

    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;

    int64_t nouter = csc ? n : m ;
    int64_t ninner = csc ? m : n ;

    const int64_t *Tile_vdim = csc ? Tile_cols : Tile_rows ;
    const int64_t *Tile_vlen = csc ? Tile_rows : Tile_cols ;

    int64_t anvec = A->nvec ;
    const int64_t *restrict Ap = A->p ;
    const int64_t *restrict Ah = A->h ;
    const int64_t *restrict Ai = A->i ;
    const bool A_iso = A->iso ;

    //--------------------------------------------------------------------------
    // allocate workspace
    //--------------------------------------------------------------------------

    size_t Wp_size = 0 ;
    int64_t *restrict Wp = NULL ;
    Wp = GB_MALLOC_WORK (anvec, int64_t, &Wp_size) ;
    if (Wp == NULL)
    { 
        // out of memory
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }
    GB_memcpy (Wp, Ap, anvec * sizeof (int64_t), nthreads_max) ;

    //--------------------------------------------------------------------------
    // split A into tiles
    //--------------------------------------------------------------------------

    int64_t akend = 0 ;

    for (int64_t outer = 0 ; outer < nouter ; outer++)
    {

        //----------------------------------------------------------------------
        // find the starting and ending vector of these tiles
        //----------------------------------------------------------------------

        // The tile appears in vectors avstart:avend-1 of A, and indices
        // aistart:aiend-1.

        const int64_t avstart = Tile_vdim [outer] ;
        const int64_t avend   = Tile_vdim [outer+1] ;
        int64_t akstart = akend ;

        if (A_is_hyper)
        { 
            // A is hypersparse: look for vector avend in the A->h hyper list.
            // The vectors to handle for this outer loop are in
            // Ah [akstart:akend-1].
            akend = akstart ;
            int64_t pright = anvec - 1 ;
            bool found ;
            GB_SPLIT_BINARY_SEARCH (avend, Ah, akend, pright, found) ;
            ASSERT (GB_IMPLIES (akstart <= akend-1, Ah [akend-1] < avend)) ;
        }
        else
        { 
            // A is sparse; the vectors to handle are akstart:akend-1
            akend = avend ;
        }

        // # of vectors in all tiles in this outer loop
        int64_t cnvec = akend - akstart ;
        int nth = GB_nthreads (cnvec, chunk, nthreads_max) ;

        //----------------------------------------------------------------------
        // create all tiles for vectors akstart:akend-1 in A
        //----------------------------------------------------------------------

        for (int64_t inner = 0 ; inner < ninner ; inner++)
        {

            //------------------------------------------------------------------
            // allocate C, C->p, and C->h for this tile
            //------------------------------------------------------------------

            const int64_t aistart = Tile_vlen [inner] ;
            const int64_t aiend   = Tile_vlen [inner+1] ;
            const int64_t cvdim = avend - avstart ;
            const int64_t cvlen = aiend - aistart ;

            C = NULL ;
            GB_OK (GB_new (&C, // new header
                atype, cvlen, cvdim, GB_Ap_malloc, csc, A_sparsity,
                hyper_switch, cnvec)) ;
            C->sparsity_control = sparsity_control ;
            C->hyper_switch = hyper_switch ;
            C->nvec = cnvec ;
            int64_t *restrict Cp = C->p ;
            int64_t *restrict Ch = C->h ;

            //------------------------------------------------------------------
            // determine the boundaries of this tile
            //------------------------------------------------------------------

            int64_t k ;
            #pragma omp parallel for num_threads(nth) schedule(static)
            for (k = akstart ; k < akend ; k++)
            {
                int64_t pA = Wp [k] ;
                const int64_t pA_end = Ap [k+1] ;
                const int64_t aknz = pA_end - pA ;
                if (aknz == 0 || Ai [pA] >= aiend)
                { 
                    // this vector of C is empty
                }
                else if (aknz > 256)
                { 
                    // use binary search to find aiend
                    bool found ;
                    int64_t pright = pA_end - 1 ;
                    GB_SPLIT_BINARY_SEARCH (aiend, Ai, pA, pright, found) ;
                    #ifdef GB_DEBUG
                    // check the results with a linear search
                    int64_t p2 = Wp [k] ;
                    for ( ; p2 < Ap [k+1] ; p2++)
                    {
                        if (Ai [p2] >= aiend) break ;
                    }
                    ASSERT (pA == p2) ;
                    #endif
                }
                else
                { 
                    // use a linear-time search to find aiend
                    for ( ; pA < pA_end ; pA++)
                    {
                        if (Ai [pA] >= aiend) break ;
                    }
                    #ifdef GB_DEBUG
                    // check the results with a binary search
                    bool found ;
                    int64_t p2 = Wp [k] ;
                    int64_t p2_end = Ap [k+1] - 1 ;
                    GB_SPLIT_BINARY_SEARCH (aiend, Ai, p2, p2_end, found) ;
                    ASSERT (pA == p2) ;
                    #endif
                }
                Cp [k-akstart] = (pA - Wp [k]) ; // # of entries in this vector
                if (A_is_hyper)
                { 
                    Ch [k-akstart] = Ah [k] - avstart ;
                }
            }

            GB_cumsum (Cp, cnvec, &(C->nvec_nonempty), nth, Werk) ;
            int64_t cnz = Cp [cnvec] ;

            //------------------------------------------------------------------
            // allocate C->i and C->x for this tile
            //------------------------------------------------------------------

            // set C->iso = A_iso       OK
            GB_OK (GB_bix_alloc (C, cnz, GxB_SPARSE, false, true, A_iso)) ;
            int64_t *restrict Ci = C->i ;
            C->nvals = cnz ;
            C->magic = GB_MAGIC ;       // for GB_nnz_held(C), to slice C

            //------------------------------------------------------------------
            // copy the tile from A into C
            //------------------------------------------------------------------

            int C_ntasks, C_nthreads ;
            GB_SLICE_MATRIX (C, 8) ;

            info = GrB_NO_VALUE ;

            if (A_iso)
            { 

                //--------------------------------------------------------------
                // split an iso matrix A into an iso tile C
                //--------------------------------------------------------------

                // A is iso and so is C; copy the iso entry
                GBURBLE ("(iso sparse split) ") ;
                memcpy (C->x, A->x, asize) ;
                #define GB_ISO_SPLIT
                #define GB_COPY(pC,pA) ;
                #include "GB_split_sparse_template.c"
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
                        #define GB_COPY(pC,pA) Cx [pC] = Ax [pA] ;

                        case GB_1BYTE : // uint8, int8, bool, or 1-byte user
                            #define GB_C_TYPE uint8_t
                            #define GB_A_TYPE uint8_t
                            #include "GB_split_sparse_template.c"
                            info = GrB_SUCCESS ;
                            break ;

                        case GB_2BYTE : // uint16, int16, or 2-byte user-defined
                            #define GB_C_TYPE uint16_t
                            #define GB_A_TYPE uint16_t
                            #include "GB_split_sparse_template.c"
                            info = GrB_SUCCESS ;
                            break ;

                        case GB_4BYTE : // uint32, int32, float, or 4-byte user
                            #define GB_C_TYPE uint32_t
                            #define GB_A_TYPE uint32_t
                            #include "GB_split_sparse_template.c"
                            info = GrB_SUCCESS ;
                            break ;

                        case GB_8BYTE : // uint64, int64, double, float complex,
                                        // or 8-byte user defined
                            #define GB_C_TYPE uint64_t
                            #define GB_A_TYPE uint64_t
                            #include "GB_split_sparse_template.c"
                            info = GrB_SUCCESS ;
                            break ;

                        case GB_16BYTE : // double complex or 16-byte user
                            #define GB_C_TYPE GB_blob16
                            #define GB_A_TYPE GB_blob16
                            #include "GB_split_sparse_template.c"
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
                ASSERT_OP_OK (op, "identity op for split sparse", GB0) ;
                info = GB_split_sparse_jit (C, op, A, akstart, aistart, Wp,
                    C_ek_slicing, C_ntasks, C_nthreads) ;
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
                #define GB_COPY(pC,pA)                          \
                    memcpy (Cx + (pC)*asize, Ax +(pA)*asize, asize) ;
                #include "GB_split_sparse_template.c"
                info = GrB_SUCCESS ;
            }

            //------------------------------------------------------------------
            // free workspace
            //------------------------------------------------------------------

            GB_WERK_POP (C_ek_slicing, int64_t) ;

            if (info != GrB_SUCCESS)
            { 
                // out of memory, or other error
                GB_FREE_ALL ;
                return (info) ;
            }

            //------------------------------------------------------------------
            // advance to the next tile
            //------------------------------------------------------------------

            if (inner < ninner - 1)
            {
                int64_t k ;
                #pragma omp parallel for num_threads(nth) schedule(static)
                for (k = akstart ; k < akend ; k++)
                { 
                    int64_t ck = k - akstart ;
                    int64_t cknz = Cp [ck+1] - Cp [ck] ;
                    Wp [k] += cknz ;
                }
            }

            //------------------------------------------------------------------
            // conform the tile and save it in the Tiles array
            //------------------------------------------------------------------

            ASSERT_MATRIX_OK (C, "C for GB_split", GB0) ;
            GB_OK (GB_hypermatrix_prune (C, Werk)) ;
            GB_OK (GB_conform (C, Werk)) ;
            if (csc)
            { 
                GB_TILE (Tiles, inner, outer) = C ;
            }
            else
            { 
                GB_TILE (Tiles, outer, inner) = C ;
            }
            ASSERT_MATRIX_OK (C, "final tile C for GB_split", GB0) ;
            C = NULL ;
        }
    }

    GB_FREE_WORKSPACE ;
    return (GrB_SUCCESS) ;
}

