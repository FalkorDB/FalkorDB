//------------------------------------------------------------------------------
// GB_concat_hyper: concatenate an array of matrices into a hypersparse matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#define GB_FREE_ALL                 \
{                                   \
    GB_FREE_MEMORY (&Wi, Wi_size) ;        \
    GB_FREE_MEMORY (&Wj, Wj_size) ;   \
    GB_FREE_MEMORY (&Wx, Wx_size) ;   \
    GB_phybix_free (C) ;            \
}

#include "concat/GB_concat.h"
#include "extractTuples/GB_extractTuples.h"

GrB_Info GB_concat_hyper            // concatenate into a hypersparse matrix
(
    GrB_Matrix C,                   // input/output matrix for results
    const bool C_iso,               // if true, construct C as iso
    const GB_void *cscalar,         // iso value of C, if C is iso 
    const int64_t cnz,              // # of entries in C
    const GrB_Matrix *Tiles,        // 2D row-major array of size m-by-n,
    const uint64_t m,
    const uint64_t n,
    const int64_t *restrict Tile_rows,  // size m+1
    const int64_t *restrict Tile_cols,  // size n+1
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // allocate triplet workspace to construct C as hypersparse
    //--------------------------------------------------------------------------

    GrB_Info info ;
    GrB_Matrix A = NULL ;
    ASSERT_MATRIX_OK (C, "C input to concat hyper", GB0) ;

    GrB_Type ctype = C->type ;
    int64_t cvlen = C->vlen ;
    int64_t cvdim = C->vdim ;
    bool csc = C->is_csc ;
    size_t csize = ctype->size ;

    GB_MDECL (Wi, , u) ; size_t Wi_size = 0 ;
    GB_MDECL (Wj, , u) ; size_t Wj_size = 0 ;
    GB_void *restrict Wx = NULL ; size_t Wx_size = 0 ;

    bool Cp_is_32, Cj_is_32, Ci_is_32 ;
    GB_determine_pji_is_32 (&Cp_is_32, &Cj_is_32, &Ci_is_32,
        GxB_HYPERSPARSE, cnz, cvlen, cvdim, Werk) ;

    float hyper_switch = C->hyper_switch ;
    float bitmap_switch = C->bitmap_switch ;
    int sparsity_control = C->sparsity_control ;

    GB_phybix_free (C) ;

    size_t cjsize = Cj_is_32 ? sizeof (uint32_t) : sizeof (uint64_t) ;
    size_t cisize = Ci_is_32 ? sizeof (uint32_t) : sizeof (uint64_t) ;

    Wi = GB_MALLOC_MEMORY (cnz, cisize, &Wi_size) ;     // becomes C->i
    Wj = GB_MALLOC_MEMORY (cnz, cjsize, &Wj_size) ;     // freed below
    if (!C_iso)
    { 
        Wx = GB_MALLOC_MEMORY (cnz, csize, &Wx_size) ;  // freed below
    }
    if (Wi == NULL || Wj == NULL || (!C_iso && Wx == NULL))
    { 
        // out of memory
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    GB_IPTR (Wi, Ci_is_32) ;
    GB_IPTR (Wj, Cj_is_32) ;

    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;

    int64_t nouter = csc ? n : m ;
    int64_t ninner = csc ? m : n ;

    //--------------------------------------------------------------------------
    // concatenate all matrices into the list of triplets
    //--------------------------------------------------------------------------

    int64_t pC = 0 ;
    for (int64_t outer = 0 ; outer < nouter ; outer++)
    {
        for (int64_t inner = 0 ; inner < ninner ; inner++)
        {

            //------------------------------------------------------------------
            // get the tile A
            //------------------------------------------------------------------

            A = csc ? GB_TILE (Tiles, inner, outer)
                    : GB_TILE (Tiles, outer, inner) ;
            ASSERT (!GB_ANY_PENDING_WORK (A)) ;

            //------------------------------------------------------------------
            // determine where to place the tile in C
            //------------------------------------------------------------------

            // The tile A appears in vectors cvstart:cvend-1 of C, and indices
            // cistart:ciend-1.

            int64_t cvstart, cistart ;
            if (csc)
            { 
                // C is held by column
                // Tiles is row-major and accessed in column order
                cvstart = Tile_cols [outer] ;
                cistart = Tile_rows [inner] ;
            }
            else
            { 
                // C is held by row
                // Tiles is row-major and accessed in row order
                cvstart = Tile_rows [outer] ;
                cistart = Tile_cols [inner] ;
            }

            //------------------------------------------------------------------
            // extract the tuples from tile A
            //------------------------------------------------------------------

            int64_t anz = GB_nnz (A) ;
            int nth = GB_nthreads (anz, chunk, nthreads_max) ;
            int64_t pA ;

            if (Cj_is_32)
            {
                if (Ci_is_32)
                {
                    #define WORK_I Wi32
                    #define WORK_J Wj32
                    #include "concat/factory/GB_concat_hyper_template.c"
                }
                else
                {
                    #define WORK_I Wi64
                    #define WORK_J Wj32
                    #include "concat/factory/GB_concat_hyper_template.c"
                }
            }
            else
            {
                if (Ci_is_32)
                {
                    #define WORK_I Wi32
                    #define WORK_J Wj64
                    #include "concat/factory/GB_concat_hyper_template.c"
                }
                else
                {
                    #define WORK_I Wi64
                    #define WORK_J Wj64
                    #include "concat/factory/GB_concat_hyper_template.c"
                }
            }

            //------------------------------------------------------------------
            // advance the tuple counter
            //------------------------------------------------------------------

            pC += anz ;
        }
    }

    //--------------------------------------------------------------------------
    // build C from the triplets
    //--------------------------------------------------------------------------

    const GB_void *S_input = NULL ;
    if (C_iso)
    { 
        S_input = cscalar ;
    }

    GB_OK (GB_builder (
        C,                      // create C using a static or dynamic header
        ctype,                  // C->type
        cvlen,                  // C->vlen
        cvdim,                  // C->vdim
        csc,                    // C->is_csc
        (void **) &Wi,          // Wi is C->i on output, or freed on error
        &Wi_size,
        (void **) &Wj,          // Wj, free on output
        &Wj_size,
        (GB_void **) &Wx,       // Wx, free on output; or NULL if C is iso
        &Wx_size,
        false,                  // tuples need to be sorted
        true,                   // no duplicates
        cnz,                    // size of Wi and Wj in # of tuples
        true,                   // is_matrix: unused
        NULL, NULL,             // original I,J tuples
        S_input,                // cscalar if C is iso, or NULL
        C_iso,                  // true if C is iso
        cnz,                    // # of tuples
        NULL,                   // no duplicates, so dup is NUL
        ctype,                  // the type of Wx (no typecasting)
        true,                   // burble is allowed
        Werk,
        Ci_is_32, Cj_is_32,     // Wi and Wj have the same integers as C->i
                                // and C->h, respectively
        Cp_is_32, Cj_is_32, Ci_is_32    // C->[pji]_is_32 for the new matrix
    )) ;

    C->hyper_switch = hyper_switch ;
    C->bitmap_switch = bitmap_switch ;
    C->sparsity_control = sparsity_control ;
    ASSERT (GB_IS_HYPERSPARSE (C)) ;
    ASSERT_MATRIX_OK (C, "C from concat hyper", GB0) ;

    // workspace has been freed by GB_builder, or transplanted into C
    ASSERT (Wi == NULL) ;
    ASSERT (Wj == NULL) ;
    ASSERT (Wx == NULL) ;

    return (GrB_SUCCESS) ;
}

