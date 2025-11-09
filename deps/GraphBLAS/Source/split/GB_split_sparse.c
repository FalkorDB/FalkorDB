//------------------------------------------------------------------------------
// GB_split_sparse: split a sparse/hypersparse matrix into tiles
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Each output tile is first created in sparse/hyper form, matching the input
// matrix, and then conformed to its desired sparsity format.

#define GB_FREE_WORKSPACE                   \
    GB_WERK_POP (C_ek_slicing, int64_t) ;   \
    GB_FREE_MEMORY (&Wp, Wp_size) ;

#define GB_FREE_ALL                         \
    GB_FREE_WORKSPACE ;                     \
    GB_Matrix_free (&C) ;

#include "split/GB_split.h"
#include "jitifyer/GB_stringify.h"
#include "apply/GB_apply.h"

GrB_Info GB_split_sparse            // split a sparse matrix
(
    GrB_Matrix *Tiles,              // 2D row-major array of size m-by-n
    const int64_t m,
    const int64_t n,
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
    ASSERT (!GB_JUMBLED (A)) ;
    ASSERT (!GB_ZOMBIES (A)) ;
    ASSERT (!GB_PENDING (A)) ;

    int sparsity_control = A->sparsity_control ;
    float hyper_switch = A->hyper_switch ;
    bool csc = A->is_csc ;
    GrB_Type atype = A->type ;
    size_t asize = atype->size ;

    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;

    int64_t nouter = csc ? n : m ;
    int64_t ninner = csc ? m : n ;

    const int64_t *Tile_vdim = csc ? Tile_cols : Tile_rows ;
    const int64_t *Tile_vlen = csc ? Tile_rows : Tile_cols ;

    int64_t anvec = A->nvec ;
    int64_t anz = GB_nnz (A) ;

    GB_Ap_DECLARE (Ap, const) ; GB_Ap_PTR (Ap, A) ;
    GB_Ah_DECLARE (Ah, const) ; GB_Ah_PTR (Ah, A) ;
    GB_Ai_DECLARE (Ai, const) ; GB_Ai_PTR (Ai, A) ;

    const bool A_iso = A->iso ;

    const bool Ap_is_32 = A->p_is_32 ;
    const bool Aj_is_32 = A->j_is_32 ;
    const bool Ai_is_32 = A->i_is_32 ;

    //--------------------------------------------------------------------------
    // allocate workspace
    //--------------------------------------------------------------------------

    // FUTURE: Wp is allocated with the same integers as Ap, but it could be
    // chosen based on anz instead.

    GB_MDECL (Wp, , u) ; size_t Wp_size = 0 ;
    size_t apsize = (Ap_is_32) ? sizeof (uint32_t) : sizeof (uint64_t) ;
    Wp = GB_MALLOC_MEMORY (anvec, apsize, &Wp_size) ;
    if (Wp == NULL)
    { 
        // out of memory
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }
    GB_memcpy (Wp, Ap, anvec * apsize, nthreads_max) ;
    GB_IPTR (Wp, Ap_is_32) ;

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
            GB_split_binary_search (avend, Ah, Aj_is_32, &akend, &pright) ;
            ASSERT (GB_IMPLIES (akstart <= akend-1,
                GB_IGET (Ah, akend-1) < avend)) ;
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

            // Assume this tile C can acquire all the entries of A to determine
            // the p_is_32, j_is_32, and i_is_32 settings for the new Tile.
            bool Cp_is_32, Cj_is_32, Ci_is_32 ;
            GB_determine_pji_is_32 (&Cp_is_32, &Cj_is_32, &Ci_is_32,
                A_sparsity, anz, cvlen, cvdim, Werk) ;

            C = NULL ;
            GB_OK (GB_new (&C, // new header
                atype, cvlen, cvdim, GB_ph_malloc, csc, A_sparsity,
                hyper_switch, cnvec, Cp_is_32, Cj_is_32, Ci_is_32)) ;
            C->sparsity_control = sparsity_control ;
            C->hyper_switch = hyper_switch ;
            C->nvec = cnvec ;

            GB_Cp_DECLARE (Cp, ) ; GB_Ap_PTR (Cp, C) ;
            GB_Ch_DECLARE (Ch, ) ; GB_Ah_PTR (Ch, C) ;

            //------------------------------------------------------------------
            // determine the boundaries of this tile
            //------------------------------------------------------------------

            int64_t k ;
            #pragma omp parallel for num_threads(nth) schedule(static)
            for (k = akstart ; k < akend ; k++)
            {
                const int64_t pC_start = GB_IGET (Wp, k) ;
                int64_t pA = pC_start ;
                const int64_t pA_end = GB_IGET (Ap, k+1) ;
                const int64_t aknz = pA_end - pA ;
                if (aknz == 0 || GB_IGET (Ai, pA) >= aiend)
                { 
                    // this vector of C is empty
                }
                else if (aknz > 256)
                { 
                    // use binary search to find aiend
                    int64_t pright = pA_end - 1 ;
                    GB_split_binary_search (aiend, Ai, Ai_is_32, &pA, &pright) ;
                    #ifdef GB_DEBUG
                    // check the results with a linear search
                    int64_t p2 = pC_start ;
                    for ( ; p2 < pA_end ; p2++)
                    {
                        if (GB_IGET (Ai, p2) >= aiend) break ;
                    }
                    ASSERT (pA == p2) ;
                    #endif
                }
                else
                { 
                    // use a linear-time search to find aiend
                    for ( ; pA < pA_end ; pA++)
                    {
                        if (GB_IGET (Ai, pA) >= aiend) break ;
                    }
                    #ifdef GB_DEBUG
                    // check the results with a binary search
                    int64_t p2 = pC_start ;
                    int64_t p2_end = pA_end - 1 ;
                    GB_split_binary_search (aiend, Ai, Ai_is_32, &p2, &p2_end) ;
                    ASSERT (pA == p2) ;
                    #endif
                }
                int64_t kC = k - akstart ;
                int64_t cknz = pA - pC_start ;      // # entries in C(:,kC)
                GB_ISET (Cp, kC, cknz) ;            // Cp [kC] = cknz ;
                if (A_is_hyper)
                { 
                    int64_t jC = GB_IGET (Ah, k) - avstart ;
                    GB_ISET (Ch, kC, jC) ;          // Ch [kC] = jC ;
                }
            }

            int64_t nvec_nonempty ;
            GB_cumsum (Cp, Cp_is_32, cnvec, &nvec_nonempty, nth, Werk) ;
            GB_nvec_nonempty_set (C, nvec_nonempty) ;
            int64_t cnz = GB_IGET (Cp, cnvec) ;

            //------------------------------------------------------------------
            // allocate C->i and C->x for this tile
            //------------------------------------------------------------------

            GB_OK (GB_bix_alloc (C, cnz, GxB_SPARSE, false, true, A_iso)) ;
            GB_Ci_DECLARE (Ci, ) ; GB_Ci_PTR (Ci, C) ;
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
                #include "split/template/GB_split_sparse_template.c"
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
                            #include "split/template/GB_split_sparse_template.c"
                            info = GrB_SUCCESS ;
                            break ;

                        case GB_2BYTE : // uint16, int16, or 2-byte user-defined
                            #define GB_C_TYPE uint16_t
                            #define GB_A_TYPE uint16_t
                            #include "split/template/GB_split_sparse_template.c"
                            info = GrB_SUCCESS ;
                            break ;

                        case GB_4BYTE : // uint32, int32, float, or 4-byte user
                            #define GB_C_TYPE uint32_t
                            #define GB_A_TYPE uint32_t
                            #include "split/template/GB_split_sparse_template.c"
                            info = GrB_SUCCESS ;
                            break ;

                        case GB_8BYTE : // uint64, int64, double, float complex,
                                        // or 8-byte user defined
                            #define GB_C_TYPE uint64_t
                            #define GB_A_TYPE uint64_t
                            #include "split/template/GB_split_sparse_template.c"
                            info = GrB_SUCCESS ;
                            break ;

                        case GB_16BYTE : // double complex or 16-byte user
                            #define GB_C_TYPE GB_blob16
                            #define GB_A_TYPE GB_blob16
                            #include "split/template/GB_split_sparse_template.c"
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
                GBURBLE ("(generic split) ") ;
                #define GB_C_TYPE GB_void
                #define GB_A_TYPE GB_void
                #undef  GB_COPY
                #define GB_COPY(pC,pA)                          \
                    memcpy (Cx + (pC)*asize, Ax +(pA)*asize, asize) ;
                #include "split/template/GB_split_sparse_template.c"
                info = GrB_SUCCESS ;
            }

            //------------------------------------------------------------------
            // free workspace
            //------------------------------------------------------------------

            GB_WERK_POP (C_ek_slicing, int64_t) ;
            GB_OK (info) ;

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
                    int64_t cknz = GB_IGET (Cp, ck+1) - GB_IGET (Cp, ck) ;
                    GB_IINC (Wp, k, cknz) ;     // Wp [k] += cknz ;
                }
            }

            //------------------------------------------------------------------
            // conform the tile and save it in the Tiles array
            //------------------------------------------------------------------

            ASSERT_MATRIX_OK (C, "C for GB_split", GB0) ;
            GB_OK (GB_hyper_prune (C, Werk)) ;
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

