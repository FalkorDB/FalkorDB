//------------------------------------------------------------------------------
// GB_matvec_check: print a GraphBLAS matrix and check if it is valid
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// in production: turn off developer flag
#define GB_DEVELOPER 0

// For development only:
// #undef  GB_DEVELOPER
// #define GB_DEVELOPER 1

#include "GB.h"
#include "pending/GB_Pending.h"
#include "get_set/GB_get_set.h"

GrB_Info GB_matvec_check    // check a GraphBLAS matrix or vector
(
    const GrB_Matrix A,     // GraphBLAS matrix to print and check
    const char *name,       // name of the matrix, optional
    int pr,                 // print level (see GB_check.h)
    FILE *f,                // file for output (or stdout if f is NULL)
    const char *kind        // "matrix" or "vector" (only for printing diag.)
)
{

    //--------------------------------------------------------------------------
    // decide what to print
    //--------------------------------------------------------------------------

    GrB_Info info ;
    bool is_hyper = GB_IS_HYPERSPARSE (A) ;
    bool is_full = GB_IS_FULL (A) ;
    bool is_bitmap = GB_IS_BITMAP (A) ;
    bool is_sparse = GB_IS_SPARSE (A) ;
    char *string = NULL ;
    size_t string_size = 0 ;

    bool skip_zombie_checks = false ;
    if (pr > 5)
    {
        pr = pr - 6 ;
        skip_zombie_checks = true ;
    }
    pr = GB_IMIN (pr, GxB_COMPLETE_VERBOSE) ;
    bool phantom = (is_full && (A->x == NULL || A->iso)) ;
    if (phantom)
    { 
        // the matrix is a phantom: iso full, or full with no A->x;
        // convert GxB_COMPLETE* to GxB_SHORT* for a phantom matrix
        if (pr == GxB_COMPLETE_VERBOSE) pr = GxB_SHORT_VERBOSE ;
        if (pr == GxB_COMPLETE        ) pr = GxB_SHORT ;
    }
//  bool pr_silent   = (pr == GxB_SILENT) ;
    bool pr_complete = (pr == GxB_COMPLETE || pr == GxB_COMPLETE_VERBOSE) ;
    bool pr_short    = (pr == GxB_SHORT    || pr == GxB_SHORT_VERBOSE   ) ;
    bool one_based = GB_Global_print_one_based_get ( ) ;
    bool pr_mem_shallow = GB_Global_stats_mem_shallow_get ( ) ;
    int64_t offset = (one_based) ? 1 : 0 ;
    #if GB_DEVELOPER
    int pr_developer = pr ;
    #else
    int pr_developer = 0 ;
    #endif

    GBPR0 ("\n  " GBd "x" GBd " GraphBLAS ",
        (A != NULL) ? GB_NROWS (A) : 0,
        (A != NULL) ? GB_NCOLS (A) : 0) ;

    if (A != NULL && A->type != NULL)
    {
        if (A->type == GxB_FC32)
        { 
            GBPR0 ("float complex") ;
        }
        else if (A->type == GxB_FC64)
        { 
            GBPR0 ("double complex") ;
        }
        else
        { 
            if (A->type->name [0] == '\0')
            { 
                GBPR0 ("user-defined (type size %zu bytes)", A->type->size) ;
            }
            else
            { 
                GBPR0 ("%s", A->type->name) ;
            }
        }
    }

    GBPR0 (" %s", kind) ;

    #if GB_DEVELOPER
    if (phantom) GBPR0 (" (phantom)") ;
    #endif

    //--------------------------------------------------------------------------
    // check if null, freed, or uninitialized
    //--------------------------------------------------------------------------

    if (A == NULL)
    { 
        GBPR0 (" NULL\n") ;
        return (GrB_NULL_POINTER) ;
    }

    GB_CHECK_MAGIC (A) ;

    GB_Ap_DECLARE  (Ap, const) ; GB_Ap_PTR  (Ap, A) ;
    GB_Ah_DECLARE  (Ah, const) ; GB_Ah_PTR  (Ah, A) ;
    GB_Ai_DECLARE  (Ai, const) ; GB_Ai_PTR  (Ai, A) ;
    GB_MDECL (A_Yp, const, u) ; GB_GET_HYPER_PTR (A_Yp, A, p) ;
    GB_MDECL (A_Yi, const, u) ; GB_GET_HYPER_PTR (A_Yi, A, i) ;
    GB_MDECL (A_Yx, const, u) ; GB_GET_HYPER_PTR (A_Yx, A, x) ;
    const int8_t *restrict Ab = A->b ;

    //--------------------------------------------------------------------------
    // print the header
    //--------------------------------------------------------------------------

    if (is_full)
    { 
        // A->p, A->h, A->i, and A->b all null
        GBPR0 (", full") ;
    }
    else if (is_bitmap)
    { 
        // A->b not null
        GBPR0 (", bitmap") ;
    }
    else if (is_sparse)
    { 
        // A->h null, A->p not null
        GBPR0 (", sparse") ;
    }
    else if (is_hyper)
    { 
        // A->h not null
        GBPR0 (", hypersparse") ;
    }
    else
    { 
        // A is not hyper, sparse, bitmap, or full
        GBPR0 (" invalid structure\n") ;
        return (GrB_INVALID_OBJECT) ;
    }
    if (A->jumbled)
    { 
        GBPR0 (" (jumbled)") ;
    }
    GBPR0 (" %s", A->is_csc ? "by col" : "by row") ;
    if (is_sparse || is_hyper || GB_DEVELOPER)
    {
        GBPR0 (", ints: %s/%s/%s",
            A->p_is_32 ? "32" : "64",
            A->j_is_32 ? "32" : "64",
            A->i_is_32 ? "32" : "64") ;
    }
    GBPR0 ("\n") ;

    int64_t nvec_nonempty = GB_nvec_nonempty_get (A) ;
    int64_t actual_nvec_nonempty = GB_nvec_nonempty (A) ;
    int64_t nnz_max = GB_nnz_max (A) ;

    #if GB_DEVELOPER
    GBPR0 ("  max # entries: " GBd "\n", nnz_max) ;
    GBPR0 ("  vlen: " GBd , A->vlen) ;
    GBPR0 ("  vdim: " GBd "\n", A->vlen) ;
    if (nvec_nonempty != -1)
    {
        GBPR0 ("  nvec_nonempty: " GBd , nvec_nonempty) ;
    }
    GBPR0 (" nvec: " GBd " plen: " GBd "\n  hyper_switch %g "
        "bitmap_switch %g\n",
        A->nvec, A->plen, A->hyper_switch, A->bitmap_switch) ;
    #endif

    switch (A->sparsity_control)
    {

        // 1
        case GxB_HYPERSPARSE : 
            GBPR0 ("  sparsity control: hypersparse only\n") ;
            break ;

        // 2
        case GxB_SPARSE : 
            GBPR0 ("  sparsity control: sparse only\n") ;
            break ;

        // 3
        case GxB_HYPERSPARSE + GxB_SPARSE : 
            GBPR0 ("  sparsity control: sparse/hypersparse\n") ;
            break ;

        // 4
        case GxB_BITMAP : 
            GBPR0 ("  sparsity control: bitmap only\n") ;
            break ;

        // 5
        case GxB_HYPERSPARSE + GxB_BITMAP : 
            GBPR0 ("  sparsity control: hypersparse/bitmap\n") ;
            break ;

        // 6
        case GxB_SPARSE + GxB_BITMAP : 
            GBPR0 ("  sparsity control: sparse/bitmap\n") ;
            break ;

        // 7
        case GxB_HYPERSPARSE + GxB_SPARSE + GxB_BITMAP : 
            GBPR0 ("  sparsity control: hypersparse/sparse/bitmap\n") ;
            break ;

        // 8
        case GxB_FULL : 
            GBPR0 ("  sparsity control: full\n") ;
            break ;

        // 9
        case GxB_HYPERSPARSE + GxB_FULL : 
            GBPR0 ("  sparsity control: hypersparse/full\n") ;
            break ;

        // 10
        case GxB_SPARSE + GxB_FULL : 
            GBPR0 ("  sparsity control: sparse/full\n") ;
            break ;

        // 11
        case GxB_HYPERSPARSE + GxB_SPARSE + GxB_FULL : 
            GBPR0 ("  sparsity control: hypersparse/sparse/full\n") ;
            break ;

        // 12
        case GxB_FULL + GxB_BITMAP : 
            GBPR0 ("  sparsity control: bitmap/full\n") ;
            break ;

        // 13
        case GxB_HYPERSPARSE + GxB_BITMAP + GxB_FULL : 
            GBPR0 ("  sparsity control: hypersparse/bitmap/full\n") ;
            break ;

        // 14
        case GxB_SPARSE + GxB_BITMAP + GxB_FULL : 
            GBPR0 ("  sparsity control: sparse/bitmap/full\n") ;
            break ;

        // 15
        case GxB_HYPERSPARSE + GxB_SPARSE + GxB_BITMAP + GxB_FULL : 
            #if GB_DEVELOPER
            GBPR0 ("  sparsity control: hyper/sparse/bitmap/full\n") ;
            #endif
            break ;

        default : 
            // invalid sparsity control
            GBPR0 ("  sparsity control: %d invalid\n", A->sparsity_control) ;
            return (GrB_INVALID_OBJECT) ;
            break ;
    }

    //--------------------------------------------------------------------------
    // check the dimensions
    //--------------------------------------------------------------------------

    if (A->vlen < 0 || A->vlen > GB_NMAX || A->vdim < 0 || A->vdim > GB_NMAX)
    { 
        GBPR0 ("  invalid %s dimensions\n", kind) ;
        return (GrB_INVALID_OBJECT) ;
    }

    //--------------------------------------------------------------------------
    // check the 32/64 bit sizes
    //--------------------------------------------------------------------------

    if (is_hyper || is_sparse)
    { 
        #if GB_DEVELOPER
        GBPR0 ("  p_control: %d, j_control: %d, i_control: %d\n",
            A->p_control, A->j_control, A->i_control) ;
        #endif
        if (!GB_valid_pji_is_32 (A->p_is_32, A->j_is_32, A->i_is_32,
            A->nvals, A->vlen, A->vdim))
        { 
            GBPR0 ("  matrix is too large for 32-bit integers\n") ;
            return (GrB_INVALID_OBJECT) ;
        }
    }

    //--------------------------------------------------------------------------
    // check vector structure
    //--------------------------------------------------------------------------

    if (is_full) 
    {
        // A is full
        if (! (A->nvec == A->vdim && A->plen == -1))
        { 
            GBPR0 ("  invalid full %s structure\n", kind) ;
            return (GrB_INVALID_OBJECT) ;
        }
    }
    else if (is_bitmap) 
    {
        // A is bitmap
        if (! (A->nvec == A->vdim && A->plen == -1 &&
               A->h == NULL && A->p == NULL && A->i == NULL))
        { 
            GBPR0 ("  invalid bitmap %s structure\n", kind) ;
            return (GrB_INVALID_OBJECT) ;
        }
    }
    else if (is_sparse)
    {
        // A is sparse
        if (! (A->nvec == A->plen && A->plen == A->vdim))
        { 
            GBPR0 ("  invalid sparse %s structure\n", kind) ;
            return (GrB_INVALID_OBJECT) ;
        }
    }
    else
    {
        // A is hypersparse
        if (! (A->nvec >= 0 && A->nvec <= A->plen && A->plen <= A->vdim))
        { 
            GBPR0 ("  invalid hypersparse %s structure\n", kind) ;
            return (GrB_INVALID_OBJECT) ;
        }
    }

    //--------------------------------------------------------------------------
    // count the allocated blocks
    //--------------------------------------------------------------------------

    int64_t nallocs ;
    size_t mem_deep, mem_shallow, memsize ;
    GB_memoryUsage (&nallocs, &mem_deep, &mem_shallow, A, true) ;
    memsize = mem_deep + (pr_mem_shallow ? mem_shallow : 0) ;

    #if GB_DEVELOPER
    if (pr_short || pr_complete)
    {
        if (A->header_size == 0)
        {
            GBPR ("  static header,") ;
        }
        else
        {
            GBPR ("  header %p", (void *) A) ;
        }
        GBPR (" number of memory blocks: " GBd "\n", nallocs) ;
        GBPR ("  deep: " GBu " shallow: " GBu " total: " GBu "\n",
            (uint64_t) mem_deep, (uint64_t) mem_shallow,
            (uint64_t) (mem_deep + mem_shallow)) ;
    }
    #endif

    //--------------------------------------------------------------------------
    // check the type
    //--------------------------------------------------------------------------

    info = GB_Type_check (A->type, "", pr_developer, f) ;
    if (info != GrB_SUCCESS)
    { 
        GBPR0 ("  %s has an invalid type\n", kind) ;
        return (GrB_INVALID_OBJECT) ;
    }

    //--------------------------------------------------------------------------
    // report shallow structure
    //--------------------------------------------------------------------------

    #if GB_DEVELOPER
    if (pr_short || pr_complete)
    {
        GBPR ("  ->h: %p shallow: %d size: " GBu "\n",
            A->h, A->h_shallow, (uint64_t) A->h_size) ;
        GBPR ("  ->p: %p shallow: %d size: " GBu "\n",
            A->p, A->p_shallow, (uint64_t) A->p_size) ;
        GBPR ("  ->i: %p shallow: %d size: " GBu "\n",
            A->i, A->i_shallow, (uint64_t) A->i_size) ;
        GBPR ("  ->b: %p shallow: %d size: " GBu "\n",
            A->b, A->b_shallow, (uint64_t) A->b_size) ;
        GBPR ("  ->x: %p shallow: %d size: " GBu "\n",
            A->x, A->x_shallow, (uint64_t) A->x_size) ;
        GBPR ("  ->Y: %p shallow: %d no_hyper_hash: %d\n",
            (void *) (A->Y), A->Y_shallow, A->no_hyper_hash) ;
    }
    #endif

    int64_t psize = A->p_is_32 ? sizeof (uint32_t) : sizeof (uint64_t) ;
    int64_t jsize = A->j_is_32 ? sizeof (uint32_t) : sizeof (uint64_t) ;
    int64_t isize = A->i_is_32 ? sizeof (uint32_t) : sizeof (uint64_t) ;

    if (A->p != NULL && (A->p_size < (A->plen + 1) * psize))
    { 
        GBPR0 ("  A->p is too small!\n") ;
        return (GrB_INVALID_OBJECT) ;
    }

    if (A->h != NULL && (A->h_size < (A->plen) * jsize))
    { 
        GBPR0 ("  A->h is too small!\n") ;
        return (GrB_INVALID_OBJECT) ;
    }

    //--------------------------------------------------------------------------
    // check p
    //--------------------------------------------------------------------------

    if (is_hyper || is_sparse)
    {
        if (A->p == NULL)
        { 
            GBPR0 ("  ->p is NULL, invalid %s\n", kind) ;
            return (GrB_INVALID_OBJECT) ;
        }
        int64_t p = GBp (Ap, 0, 0) ;
        if (p != 0)
        { 
            GBPR0 ("  ->p [0] = " GBd " invalid\n", p) ;
            return (GrB_INVALID_OBJECT) ;
        }
        int64_t nzmax = A->i_size / isize ;
        for (int64_t j = 0 ; j < A->nvec ; j++)
        {
            int64_t p    = GBp (Ap, j,   0) ;
            int64_t pend = GBp (Ap, j+1, 0) ;
            if (pend < p || pend > nzmax)
            { 
                GBPR0 ("  ->p [" GBd "] = " GBd " invalid\n", j+1, pend) ;
                return (GrB_INVALID_OBJECT) ;
            }
        }
        if (nzmax > 0 && A->i == NULL)
        { 
            GBPR0 ("  ->i is NULL, invalid %s\n", kind) ;
            return (GrB_INVALID_OBJECT) ;
        }
    }

    //--------------------------------------------------------------------------
    // check the content of h
    //--------------------------------------------------------------------------

    if (is_hyper)
    {
        int64_t jlast = -1 ;
        for (int64_t k = 0 ; k < A->nvec ; k++)
        {
            int64_t j = GBh (Ah, k) ;
            if (jlast >= j || j < 0 || j >= A->vdim)
            { 
                GBPR0 ("  ->h [" GBd "] = " GBd " invalid\n", k, j) ;
                return (GrB_INVALID_OBJECT) ;
            }
            jlast = j ;
        }
    }

    //--------------------------------------------------------------------------
    // report name and number of entries
    //--------------------------------------------------------------------------

    GBPR0 ("  ") ;
    if (name != NULL && strlen (name) > 0)
    { 
        GBPR0 ("%s, ", ((name != NULL) ? name : "")) ;
    }

    // # of entries cannot be computed until all the tests above are OK
    int64_t anz = GB_nnz (A) ;
    if (anz == 0)
    { 
        GBPR0 ("no entries") ;
    }
    else if (anz == 1)
    { 
        GBPR0 ("1 entry") ;
    }
    else if (anz == INT64_MAX)
    { 
        // GB_nnz (A) resulted in integer overflow, so use double instead
        GBPR0 ("%g entries", ((double) A->vlen) * ((double) A->vdim)) ;
    }
    else
    { 
        GBPR0 (GBd " entries", anz) ;
    }

    //--------------------------------------------------------------------------
    // print the memory size
    //--------------------------------------------------------------------------

    #define K (1024L)
    if (memsize < K)
    { 
        GBPR0 (", memory: " GBd " bytes\n", (int64_t) memsize) ;
    }
    else if (memsize < K*K)
    { 
        double s = ((double) memsize) / ((double) K) ;
        GBPR0 (", memory: %.1f KB\n", s) ;
    }
    else if (memsize < K*K*K)
    { 
        double s = ((double) memsize) / ((double) K*K) ;
        GBPR0 (", memory: %.1f MB\n", s) ;
    }
    else
    {
        double s = ((double) memsize) / ((double) K*K*K) ;
        GBPR0 (", memory: %.1f GB\n", s) ;
    }

    //--------------------------------------------------------------------------
    // print the iso value
    //--------------------------------------------------------------------------

    if (A->iso)
    {
        if (A->x == NULL || A->x_size < A->type->size)
        { 
            #if GB_DEVELOPER
            GBPR0 ("A->x %p size %d typesize %d\n", A->x, (int) A->x_size,
                (int) A->type->size) ;
            #endif
            GBPR0 ("  invalid iso matrix\n") ;
            return (GrB_INVALID_OBJECT) ;
        }
        GBPR0 ("  iso value: ") ;
        if (pr > 0)
        { 
            info = GB_entry_check (A->type, A->x, pr, f, &string, &string_size) ;
            if (info != GrB_SUCCESS)
            { 
                GB_FREE_MEMORY (&string, string_size) ;
                return (info) ;
            }
        }
        GBPR0 ("\n") ;
    }

    //--------------------------------------------------------------------------
    // check the # of entries
    //--------------------------------------------------------------------------

    if (!(A->iso) && anz > nnz_max)
    {
        GBPR0 ("\nentries: " GBd " max entries: " GBd " (invalid)\n",
            anz, nnz_max) ;
        GB_FREE_MEMORY (&string, string_size) ;
        return (GrB_INVALID_OBJECT) ;
    }

    //--------------------------------------------------------------------------
    // report the GrB_set name
    //--------------------------------------------------------------------------

    // name given by GrB_set
    char *given_name = A->user_name ;
    if (A->user_name_size > 0 && given_name != NULL)
    { 
        GBPR0 ("    %s given name: [%s]\n", kind, given_name) ;
    }

    //--------------------------------------------------------------------------
    // report the number of pending tuples and zombies
    //--------------------------------------------------------------------------

    GB_Pending Pending = A->Pending ;

    if (Pending != NULL || A->nzombies != 0)
    { 
        GBPR0 ("  pending tuples: " GBd " max pending: " GBd 
            " zombies: " GBd "\n", GB_Pending_n (A),
            (Pending == NULL) ? 0 : (Pending->nmax),
            A->nzombies) ;
    }

    if (is_full || is_bitmap)
    {
        if (A->nzombies != 0)
        { 
            // full/bitmap cannot have zombies
            GBPR0 ("  %s %s cannot have zombies\n",
                is_full ? "full" : "bitmap", kind) ;
            GB_FREE_MEMORY (&string, string_size) ;
            return (GrB_INVALID_OBJECT) ;
        }
        if (Pending != NULL)
        { 
            // full/bitmap cannot have pending tuples
            GBPR0 ("  %s %s cannot have pending tuples\n",
                is_full ? "full" : "bitmap", kind) ;
            GB_FREE_MEMORY (&string, string_size) ;
            return (GrB_INVALID_OBJECT) ;
        }
        if (A->jumbled)
        { 
            // full/bitmap jumbled
            GBPR0 ("  %s %s cannot be jumbled\n",
                is_full ? "full" : "bitmap", kind) ;
            GB_FREE_MEMORY (&string, string_size) ;
            return (GrB_INVALID_OBJECT) ;
        }
    }

    if (!skip_zombie_checks)
    {
        if (A->nzombies > anz)
        { 
            GBPR0 ("  invalid number of zombies: " GBd " "
                "must be >= 0 and <= # entries (" GBd ")\n", A->nzombies, anz) ;
            GB_FREE_MEMORY (&string, string_size) ;
            return (GrB_INVALID_OBJECT) ;
        }
    }

    //--------------------------------------------------------------------------
    // check and print the row indices and numerical values
    //--------------------------------------------------------------------------

    if (anz > 0) GBPR0 ("\n") ;

    #define GB_NBRIEF 10
    #define GB_NZBRIEF 30

    int64_t nzombies = 0 ;
    int64_t icount = 0 ;
    bool truncated = false ;
    int64_t anz_actual = 0 ;

    // for each vector of A
    for (int64_t k = 0 ; k < A->nvec ; k++)
    {
        int64_t ilast = -1 ;
        int64_t j = GBh (Ah, k) ;
        int64_t p = GBp (Ap, k, A->vlen) ;
        int64_t pend = GBp (Ap, k+1, A->vlen) ;

        // count the entries in A(:,j)
        int64_t ajnz = pend - p ;
        if (is_bitmap)
        {
            ajnz = 0 ;
            for (int64_t p2 = p ; p2 < pend ; p2++)
            {
                int8_t ab = Ab [p2] ;
                if (ab < 0 || ab > 1)
                { 
                    // bitmap with value other than 0, 1
                    GBPR0 ("    invalid bitmap %d\n", ab) ;
                    GB_FREE_MEMORY (&string, string_size) ;
                    return (GrB_INVALID_OBJECT) ;
                }
                ajnz += (ab != 0)  ;
            }
        }

        bool prcol = ((pr_short && !truncated) || pr_complete) ;
        // print the header for vector j
        if (prcol)
        { 
            #if GB_DEVELOPER
            GBPR ("  %s: " GBd " : " GBd " entries [" GBd ":" GBd "]\n",
                A->is_csc ? "column" : "row", j, ajnz, p, pend-1) ;
            if (pr_short && k == GB_NBRIEF) truncated = true ;
            #endif
        }

        // for each entry in A(:,j), the kth vector of A
        for ( ; p < pend ; p++)
        {
            if (!GBb (Ab, p)) continue ;
            anz_actual++ ;
            icount++ ;

            int64_t i = GBi (Ai, p, A->vlen) ;
            bool is_zombie = GB_IS_ZOMBIE (i) ;
            i = GB_UNZOMBIE (i) ;
            if (is_zombie) nzombies++ ;
            bool print_value = false ;
            if (prcol)
            {
                if ((pr_short && icount < GB_NZBRIEF) || pr_complete)
                { 
                    print_value = true ;
                    #if GB_DEVELOPER
                    GBPR ("    %s " GBd ": ", A->is_csc ? "row":"column", i) ;
                    #else
                    if (A->is_csc)
                    {
                        GBPR ("    (" GBd "," GBd ") ", i+offset, j+offset) ;
                    }
                    else
                    {
                        GBPR ("    (" GBd "," GBd ") ", j+offset, i+offset) ;
                    }
                    #endif
                }
                else if (pr_short && (ilast == -1 || icount == GB_NZBRIEF))
                { 
                    truncated = true ;
                }
            }
            int64_t row = A->is_csc ? i : j ;
            int64_t col = A->is_csc ? j : i ;
            if (i < 0 || i >= A->vlen)
            { 
                GBPR0 ("  index (" GBd "," GBd ") out of range\n",
                    row+offset, col+offset) ;
                GB_FREE_MEMORY (&string, string_size) ;
                return (GrB_INVALID_OBJECT) ;
            }

            // print the value
            if (print_value)
            {
                if (is_zombie)
                { 
                    GBPR ("zombie") ;
                }
                else if (A->x != NULL)
                { 
                    GB_void *Ax = (GB_void *) A->x ;
                    info = GB_entry_check (A->type,
                        Ax + (A->iso ? 0 : (p * (A->type->size))), pr, f,
                        &string, &string_size) ;
                    if (info != GrB_SUCCESS)
                    { 
                        GB_FREE_MEMORY (&string, string_size) ;
                        return (info) ;
                    }
                }
            }

            // If the matrix is known to be jumbled, then out-of-order
            // indices are OK (but duplicates are not OK).  If the matrix is
            // unjumbled, then all indices must appear in ascending order.
            if (A->jumbled ? (i == ilast) : (i <= ilast))
            { 
                // indices unsorted, or duplicates present
                GBPR0 (" index (" GBd "," GBd ") invalid\n",
                    row+offset, col+offset) ;
                GB_FREE_MEMORY (&string, string_size) ;
                return (GrB_INDEX_OUT_OF_BOUNDS) ;
            }

            if (print_value)
            { 
                GBPR ("\n") ;
            }
            ilast = i ;

            if (phantom && anz_actual >= GB_NZBRIEF)
            { 
                truncated = true ;
                break ;
            }
        }

        if (phantom && (truncated || k >= GB_NBRIEF))
        { 
            truncated = true ;
            break ;
        }
    }

    if (pr_short && truncated) GBPR ("    ...\n") ;

    //--------------------------------------------------------------------------
    // check the entry count in the bitmap
    //--------------------------------------------------------------------------

    if (is_bitmap && anz != anz_actual)
    { 
        // bitmap with invalid nvals
        GBPR0 ("  invalid bitmap count: " GBd " exist but"
            " A->nvals = " GBd "\n", anz_actual, anz) ;
        GB_FREE_MEMORY (&string, string_size) ;
        return (GrB_INVALID_OBJECT) ;
    }
    else if ((is_sparse || is_hyper) && anz != anz_actual)
    { 
        // sparse/hypersparse with invalid nvals
        GBPR0 ("  invalid sparse/hypersparse entry count: " GBd " exist but"
            " A->nvals = " GBd "\n", anz_actual, anz) ;
        GB_FREE_MEMORY (&string, string_size) ;
        return (GrB_INVALID_OBJECT) ;
    }

    //--------------------------------------------------------------------------
    // check the zombie count
    //--------------------------------------------------------------------------

    if (!skip_zombie_checks && nzombies != A->nzombies)
    { 
        GBPR0 ("  invalid zombie count: " GBd " exist but"
            " A->nzombies = " GBd "\n", nzombies, A->nzombies) ;
        GB_FREE_MEMORY (&string, string_size) ;
        return (GrB_INVALID_OBJECT) ;
    }

    //--------------------------------------------------------------------------
    // check and print the pending tuples
    //--------------------------------------------------------------------------

    #if GB_DEVELOPER
    if ((pr_short || pr_complete) && (is_sparse || is_hyper))
    {
        GBPR ("  Pending %p\n", (void *) Pending) ;
    }
    #endif

    if (Pending != NULL)
    {

        //---------------------------------------------------------------------
        // A has pending tuples
        //---------------------------------------------------------------------

        GB_MDECL (Pending_i, , u) ; GB_GET_PENDINGi_PTR (Pending_i, A) ;
        GB_MDECL (Pending_j, , u) ; GB_GET_PENDINGj_PTR (Pending_j, A) ;
        GB_void *Pending_x = Pending->x ;

        #if GB_DEVELOPER
        if (pr_short || pr_complete)
        {
            GBPR ("  Pending_i %p\n", Pending_i) ;
            GBPR ("  Pending_j %p\n", Pending_j) ;
            GBPR ("  Pending_x %p\n", Pending_x) ;
        }
        #endif

        if (Pending->n < 0 || Pending->n > Pending->nmax ||
            Pending->nmax < 0)
        { 
            GBPR0 ("  invalid pending count\n") ;
            GB_FREE_MEMORY (&string, string_size) ;
            return (GrB_INVALID_OBJECT) ;
        }

        // matrix has tuples, arrays and type must not be NULL
        // Pending_x must be NULL if and only if A is iso
        // Pending_x must be non-NULL if and only if A is non-iso
        if (Pending_i == NULL || ((Pending_x == NULL) != (A->iso)) ||
            (A->vdim > 1 && Pending_j == NULL))
        { 
            GBPR0 ("  invalid pending tuples\n") ;
            GB_FREE_MEMORY (&string, string_size) ;
            return (GrB_INVALID_OBJECT) ;
        }

        GBPR0 ("  pending tuples:\n") ;

        info = GB_Type_check (Pending->type, "", pr, f) ;
        if (info != GrB_SUCCESS || (Pending->type->size != Pending->size))
        { 
            // invalid Pending->type
            GBPR0 ("  %s has an invalid Pending->type\n", kind) ;
            GB_FREE_MEMORY (&string, string_size) ;
            return (GrB_INVALID_OBJECT) ;
        }

        int64_t ilast = -1 ;
        int64_t jlast = -1 ;
        bool sorted = true ;

        for (int64_t k = 0 ; k < Pending->n ; k++)
        {
            int64_t i = GB_IGET (Pending_i, k) ;
            int64_t j = (A->vdim <= 1) ? 0 : GB_IGET (Pending_j, k) ;
            int64_t row = (A->is_csc ? i : j) + offset ;
            int64_t col = (A->is_csc ? j : i) + offset ;

            // print the tuple
            if ((pr_short && k < GB_NZBRIEF) || pr_complete)
            {
                GBPR ("    row: " GBd " col: " GBd " ", row, col) ;
                if (!A->iso)
                { 
                    info = GB_entry_check (Pending->type,
                        Pending_x +(k * Pending->type->size), pr, f,
                        &string, &string_size) ;
                    if (info != GrB_SUCCESS)
                    { 
                        GB_FREE_MEMORY (&string, string_size) ;
                        return (info) ;
                    }
                }
                GBPR ("\n") ;
            }

            if (i < 0 || i >= A->vlen || j < 0 || j >= A->vdim)
            { 
                GBPR0 ("    tuple (" GBd "," GBd ") out of range\n", row, col) ;
                GB_FREE_MEMORY (&string, string_size) ;
                return (GrB_INVALID_OBJECT) ;
            }

            sorted = sorted && ((jlast < j) || (jlast == j && ilast <= i)) ;
            ilast = i ;
            jlast = j ;
        }

        if (sorted != Pending->sorted)
        { 
            GBPR0 ("  invalid pending tuples: invalid sort\n") ;
            GB_FREE_MEMORY (&string, string_size) ;
            return (GrB_INVALID_OBJECT) ;
        }

        if (Pending->op == NULL)
        { 
            GBPR0 ("  pending operator: implicit 2nd\n") ;
        }
        else
        {
            info = GB_BinaryOp_check (Pending->op, "pending operator:", pr, f) ;
            if (info != GrB_SUCCESS)
            { 
                GBPR0 ("  invalid pending operator\n") ;
                GB_FREE_MEMORY (&string, string_size) ;
                return (GrB_INVALID_OBJECT) ;
            }
        }
    }

    if (pr_complete)
    { 
        GBPR ("\n") ;
    }

    //--------------------------------------------------------------------------
    // check nvec_nonempty
    //--------------------------------------------------------------------------

    // A->nvec_nonempty == -1 denotes that the value has not been computed.
    // This is valid, and can occur for imported matrices and in other cases
    // when its computation is postponed or not needed.  If not -1, however,
    // the value must be correct.

    if (! ((nvec_nonempty == actual_nvec_nonempty) || (nvec_nonempty == -1)))
    { 
        // invalid nvec_nonempty
        GBPR0 ("  invalid count of non-empty vectors\n") ;
        GB_FREE_MEMORY (&string, string_size) ;
        return (GrB_INVALID_OBJECT) ;
    }

    //--------------------------------------------------------------------------
    // check A->Y
    //--------------------------------------------------------------------------

    GrB_Matrix Y = A->Y ;
    if (Y != NULL)
    {
        if (!is_hyper)
        { 
            // A->Y is optional, but A must be hypersparse for A->Y to exist
            GBPR0 ("  hyper_hash invalid\n") ;
            GB_FREE_MEMORY (&string, string_size) ;
            return (GrB_INVALID_OBJECT) ;
        }
        info = GB_matvec_check (Y, "Y hyper_hash", pr_developer, f, "matrix") ;
        if (info != GrB_SUCCESS)
        { 
            // A->Y fails the tests in GB_matvec_check
            GBPR0 ("  hyper_hash invalid") ;
            GB_FREE_MEMORY (&string, string_size) ;
            return (info) ;
        }
        GrB_Type ytype = (A->j_is_32) ? GrB_UINT32 : GrB_UINT64 ;
        if (Y->vlen != A->vdim || !GB_IS_POWER_OF_TWO (Y->vdim) ||
            Y->nvals != A->nvec || !GB_IS_SPARSE (Y) || Y->type != ytype ||
            !Y->is_csc || GB_ANY_PENDING_WORK (Y) ||
            Y->p_is_32 != A->j_is_32 || Y->i_is_32 != A->j_is_32)
        { 
            // Y must be sparse, uint32/uint64 (depending on A->j_is_32), held
            // by column, with A->nvec values, vector length the same as
            // A->vdim, and with a Y->vdim that is a power of 2. It cannot have
            // any pending work.
            GBPR0 ("  hyper_hash invalid") ;
            GB_FREE_MEMORY (&string, string_size) ;
            return (GrB_INVALID_OBJECT) ;
        }
        // ensure that Y is the inverse of A->h
        int64_t hash_bits = Y->vdim - 1 ;
        for (int64_t k = 0 ; k < A->nvec ; k++)
        {
            // look for j in the hyper_hash; it must be at position k
            int64_t j = GBh (Ah, k) ;
            int64_t jhash = GB_HASHF2 (j, hash_bits) ;
            bool found = false ;
            int64_t p = GBp (A_Yp, jhash, 0) ;
            int64_t pend = GBp (A_Yp, jhash+1, 0) ;
            for ( ; p < pend ; p++)
            {
                if (j == GBi (A_Yi, p, Y->vlen))
                {
                    if (k != GB_IGET (A_Yx, p))
                    { 
                        // j is found but not with the right value of k
                        GBPR0 ("  hyper_hash invalid\n") ;
                        GB_FREE_MEMORY (&string, string_size) ;
                        return (GrB_INVALID_OBJECT) ;
                    }
                    found = true ;
                    break ;
                }
            }
            if (!found)
            { 
                // j must appear in the hyper_hash
                GBPR0 ("  hyper_hash invalid\n") ;
                GB_FREE_MEMORY (&string, string_size) ;
                return (GrB_INVALID_OBJECT) ;
            }
        }
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    GB_FREE_MEMORY (&string, string_size) ;
    return (GrB_SUCCESS) ;
}

