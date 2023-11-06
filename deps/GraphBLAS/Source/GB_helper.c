//------------------------------------------------------------------------------
// GB_helper.c: helper functions for @GrB interface
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// JIT: not needed.  Only one variant possible.

// These functions are only used by the @GrB interface for
// SuiteSparse:GraphBLAS.

#include "GB_helper.h"

bool GB_factory_kernels_enabled = true ;

//------------------------------------------------------------------------------
// GB_NTHREADS_HELPER: determine the number of threads to use
//------------------------------------------------------------------------------

#define GB_NTHREADS_HELPER(work)                                \
    int nthreads_max = GB_Context_nthreads_max ( ) ;            \
    double chunk = GB_Context_chunk ( ) ;                       \
    int nthreads = GB_nthreads (work, chunk, nthreads_max) ;

//------------------------------------------------------------------------------
// GB_ALLOCATE_WORK: allocate per-thread workspace
//------------------------------------------------------------------------------

#define GB_ALLOCATE_WORK(work_type)                                         \
    size_t Work_size ;                                                      \
    work_type *Work = GB_MALLOC_WORK (nthreads, work_type, &Work_size) ;    \
    if (Work == NULL) return (false) ;

//------------------------------------------------------------------------------
// GB_FREE_WORKSPACE: free per-thread workspace
//------------------------------------------------------------------------------

#define GB_FREE_WORKSPACE                                                   \
    GB_FREE_WORK (&Work, Work_size) ;

//------------------------------------------------------------------------------
// GB_helper0: get the current wall-clock time from OpenMP
//------------------------------------------------------------------------------

double GB_helper0 (void)
{
    return (GB_OPENMP_GET_WTIME) ;
}

//------------------------------------------------------------------------------
// GB_helper1: convert 0-based indices to 1-based for gbextracttuples
//------------------------------------------------------------------------------

void GB_helper1              // convert zero-based indices to one-based
(
    double *restrict I_double,      // output array
    const GrB_Index *restrict I,    // input array
    int64_t nvals                   // size of input and output arrays
)
{

    GB_NTHREADS_HELPER (nvals) ;

    int64_t k ;
    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (k = 0 ; k < nvals ; k++)
    {
        I_double [k] = (double) (I [k] + 1) ;
    }
}

//------------------------------------------------------------------------------
// GB_helper1i: convert 0-based indices to 1-based for gbextracttuples
//------------------------------------------------------------------------------

void GB_helper1i             // convert zero-based indices to one-based
(
    int64_t *restrict I,            // input/output array
    int64_t nvals                   // size of input/output array
)
{

    GB_NTHREADS_HELPER (nvals) ;

    int64_t k ;
    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (k = 0 ; k < nvals ; k++)
    {
        I [k] ++ ;
    }
}

//------------------------------------------------------------------------------
// GB_helper3: convert 1-based indices to 0-based for gb_mxarray_to_list
//------------------------------------------------------------------------------

bool GB_helper3             // return true if OK, false on error
(
    int64_t *restrict List,             // size len, output array
    const double *restrict List_double, // size len, input array
    int64_t len,
    int64_t *List_max       // also compute the max entry in the list (1-based)
)
{

    GB_NTHREADS_HELPER (len) ;

    ASSERT (List != NULL) ;
    ASSERT (List_double != NULL) ;
    ASSERT (List_max != NULL) ;

    bool ok = true ;
    int64_t listmax = -1 ;

    GB_ALLOCATE_WORK (int64_t) ;

    int tid ;
    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (tid = 0 ; tid < nthreads ; tid++)
    {
        bool my_ok = true ;
        int64_t k1, k2, my_listmax = -1 ;
        GB_PARTITION (k1, k2, len, tid, nthreads) ;
        for (int64_t k = k1 ; k < k2 ; k++)
        {
            double x = List_double [k] ;
            int64_t i = (int64_t) x ;
            my_ok = my_ok && (x == (double) i) ;
            my_listmax = GB_IMAX (my_listmax, i) ;
            List [k] = i - 1 ;
        }
        // rather than create a separate per-thread boolean workspace, just
        // use a sentinal value of INT64_MIN if non-integer indices appear
        // in List_double.
        Work [tid] = my_ok ? my_listmax : INT64_MIN ;
    }

    // wrapup
    for (tid = 0 ; tid < nthreads ; tid++)
    {
        listmax = GB_IMAX (listmax, Work [tid]) ;
        ok = ok && (Work [tid] != INT64_MIN) ;
    }

    GB_FREE_WORKSPACE ;

    (*List_max) = listmax ;
    return (ok) ;
}

//------------------------------------------------------------------------------
// GB_helper3i: convert 1-based indices to 0-based for gb_mxarray_to_list
//------------------------------------------------------------------------------

bool GB_helper3i        // return true if OK, false on error
(
    int64_t *restrict List,             // size len, output array
    const int64_t *restrict List_int64, // size len, input array
    int64_t len,
    int64_t *List_max   // also compute the max entry in the list (1-based)
)
{

    GB_NTHREADS_HELPER (len) ;

    int64_t listmax = -1 ;

    GB_ALLOCATE_WORK (int64_t) ;

    int tid ;
    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (tid = 0 ; tid < nthreads ; tid++)
    {
        int64_t k1, k2, my_listmax = -1 ;
        GB_PARTITION (k1, k2, len, tid, nthreads) ;
        for (int64_t k = k1 ; k < k2 ; k++)
        {
            int64_t i = List_int64 [k] ;
            my_listmax = GB_IMAX (my_listmax, i) ;
            List [k] = i - 1 ;
        }
        Work [tid] = my_listmax ;
    }

    // wrapup
    for (tid = 0 ; tid < nthreads ; tid++)
    {
        listmax = GB_IMAX (listmax, Work [tid]) ;
    }

    GB_FREE_WORKSPACE ;

    (*List_max) = listmax ;
    return (true) ;
}

//------------------------------------------------------------------------------
// GB_helper4: find the max entry in a list of type GrB_Index
//------------------------------------------------------------------------------

bool GB_helper4             // return true if OK, false on error
(
    const GrB_Index *restrict I,    // array of size len
    const int64_t len,
    GrB_Index *List_max     // also compute the max entry in the list (1-based,
                            // which is max(I)+1)
)
{

    GB_NTHREADS_HELPER (len) ;

    GrB_Index listmax = 0 ;

    GB_ALLOCATE_WORK (GrB_Index) ;

    int tid ;
    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (tid = 0 ; tid < nthreads ; tid++)
    {
        int64_t k1, k2 ;
        GrB_Index my_listmax = 0 ;
        GB_PARTITION (k1, k2, len, tid, nthreads) ;
        for (int64_t k = k1 ; k < k2 ; k++)
        {
            my_listmax = GB_IMAX (my_listmax, I [k]) ;
        }
        Work [tid] = my_listmax ;
    }

    // wrapup
    for (tid = 0 ; tid < nthreads ; tid++)
    {
        listmax = GB_IMAX (listmax, Work [tid]) ;
    }

    GB_FREE_WORKSPACE ;

    if (len > 0) listmax++ ;
    (*List_max) = listmax ;
    return (true) ;
}

//------------------------------------------------------------------------------
// GB_helper5: construct pattern of S for gblogassign
//------------------------------------------------------------------------------

void GB_helper5              // construct pattern of S
(
    GrB_Index *restrict Si,         // array of size anz
    GrB_Index *restrict Sj,         // array of size anz
    const GrB_Index *restrict Mi,   // array of size mnz, M->i, may be NULL
    const GrB_Index *restrict Mj,   // array of size mnz,
    const int64_t mvlen,            // M->vlen
    GrB_Index *restrict Ai,         // array of size anz, A->i, may be NULL
    const int64_t avlen,            // M->vlen
    const GrB_Index anz
)
{

    GB_NTHREADS_HELPER (anz) ;
    ASSERT (Mj != NULL) ;
    ASSERT (Si != NULL) ;
    ASSERT (Sj != NULL) ;

    int64_t k ;
    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (k = 0 ; k < anz ; k++)
    {
        int64_t i = GBI (Ai, k, avlen) ;
        Si [k] = GBI (Mi, i, mvlen) ;
        Sj [k] = Mj [i] ;
    }
}

//------------------------------------------------------------------------------
// GB_helper7: Kx = uint64 (0:mnz-1), for gblogextract
//------------------------------------------------------------------------------

// TODO: use GrB_apply with a positional operator instead

void GB_helper7              // Kx = uint64 (0:mnz-1)
(
    uint64_t *restrict Kx,          // array of size mnz
    const GrB_Index mnz
)
{

    GB_NTHREADS_HELPER (mnz) ;

    int64_t k ;
    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (k = 0 ; k < mnz ; k++)
    {
        Kx [k] = k ;
    }
}

//------------------------------------------------------------------------------
// GB_helper8: expand a scalar into an array for gbbuild
//------------------------------------------------------------------------------

// TODO: use GrB_assign instead

void GB_helper8
(
    GB_void *C,         // output array of size nvals * s
    GB_void *A,         // input scalar of size s
    GrB_Index nvals,    // size of C
    size_t s            // size of each scalar
)
{

    GB_NTHREADS_HELPER (nvals) ;

    int64_t k ;
    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (k = 0 ; k < nvals ; k++)
    {
        // C [k] = A [0]
        memcpy (C + k * s, A, s) ;
    }
}

//------------------------------------------------------------------------------
// GB_helper10: compute norm (x-y,p) of two dense FP32 or FP64 vectors
//------------------------------------------------------------------------------

// p can be:

//      0 or 2:     2-norm, sqrt (sum ((x-y).^2))
//      1:          1-norm, sum (abs (x-y))
//      INT64_MAX   inf-norm, max (abs (x-y))
//      INT64_MIN   (-inf)-norm, min (abs (x-y))
//      other:      p-norm not yet computed

double GB_helper10       // norm (x-y,p), or -1 on error
(
    GB_void *x_arg,             // float or double, depending on type parameter
    bool x_iso,                 // true if x is iso
    GB_void *y_arg,             // same type as x, treat as zero if NULL
    bool y_iso,                 // true if x is iso
    GrB_Type type,              // GrB_FP32 or GrB_FP64
    int64_t p,                  // 0, 1, 2, INT64_MIN, or INT64_MAX
    GrB_Index n
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (!(type == GrB_FP32 || type == GrB_FP64))
    {
        // type of x and y must be GrB_FP32 or GrB_FP64
        return ((double) -1) ;
    }

    if (n == 0)
    {
        return ((double) 0) ;
    }

    //--------------------------------------------------------------------------
    // allocate workspace and determine # of threads to use
    //--------------------------------------------------------------------------

    GB_NTHREADS_HELPER (n) ;
    GB_ALLOCATE_WORK (double) ;

    #define xx(k) x [x_iso ? 0 : k]
    #define yy(k) y [y_iso ? 0 : k]

    //--------------------------------------------------------------------------
    // each thread computes its partial norm
    //--------------------------------------------------------------------------

    int tid ;
    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (tid = 0 ; tid < nthreads ; tid++)
    {
        int64_t k1, k2 ;
        GB_PARTITION (k1, k2, n, tid, nthreads) ;

        if (type == GrB_FP32)
        {

            //------------------------------------------------------------------
            // FP32 case
            //------------------------------------------------------------------

            float my_s = 0 ;
            const float *x = (float *) x_arg ;
            const float *y = (float *) y_arg ;
            switch (p)
            {
                case 0:     // Frobenius norm
                case 2:     // 2-norm: sqrt of sum of (x-y).^2
                {
                    if (y == NULL)
                    {
                        for (int64_t k = k1 ; k < k2 ; k++)
                        {
                            float t = xx (k) ;
                            my_s += (t*t) ;
                        }
                    }
                    else
                    {
                        for (int64_t k = k1 ; k < k2 ; k++)
                        {
                            float t = (xx (k) - yy (k)) ;
                            my_s += (t*t) ;
                        }
                    }
                }
                break ;

                case 1:     // 1-norm: sum (abs (x-y))
                {
                    if (y == NULL)
                    {
                        for (int64_t k = k1 ; k < k2 ; k++)
                        {
                            my_s += fabsf (xx (k)) ;
                        }
                    }
                    else
                    {
                        for (int64_t k = k1 ; k < k2 ; k++)
                        {
                            my_s += fabsf (xx (k) - yy (k)) ;
                        }
                    }
                }
                break ;

                case INT64_MAX:     // inf-norm: max (abs (x-y))
                {
                    if (y == NULL)
                    {
                        for (int64_t k = k1 ; k < k2 ; k++)
                        {
                            my_s = fmaxf (my_s, fabsf (xx (k))) ;
                        }
                    }
                    else
                    {
                        for (int64_t k = k1 ; k < k2 ; k++)
                        {
                            my_s = fmaxf (my_s, fabsf (xx (k) - yy (k))) ;
                        }
                    }
                }
                break ;

                case INT64_MIN:     // (-inf)-norm: min (abs (x-y))
                {
                    my_s = INFINITY ;
                    if (y == NULL)
                    {
                        for (int64_t k = k1 ; k < k2 ; k++)
                        {
                            my_s = fminf (my_s, fabsf (xx (k))) ;
                        }
                    }
                    else
                    {
                        for (int64_t k = k1 ; k < k2 ; k++)
                        {
                            my_s = fminf (my_s, fabsf (xx (k) - yy (k))) ;
                        }
                    }
                }
                break ;

                default: ;  // p-norm not yet supported
            }
            Work [tid] = (double) my_s ;

        }
        else
        {

            //------------------------------------------------------------------
            // FP64 case
            //------------------------------------------------------------------

            double my_s = 0 ;
            const double *x = (double *) x_arg ;
            const double *y = (double *) y_arg ;
            switch (p)
            {
                case 0:     // Frobenius norm
                case 2:     // 2-norm: sqrt of sum of (x-y).^2
                {
                    if (y == NULL)
                    {
                        for (int64_t k = k1 ; k < k2 ; k++)
                        {
                            double t = xx (k) ;
                            my_s += (t*t) ;
                        }
                    }
                    else
                    {
                        for (int64_t k = k1 ; k < k2 ; k++)
                        {
                            double t = (xx (k) - yy (k)) ;
                            my_s += (t*t) ;
                        }
                    }
                }
                break ;

                case 1:     // 1-norm: sum (abs (x-y))
                {
                    if (y == NULL)
                    {
                        for (int64_t k = k1 ; k < k2 ; k++)
                        {
                            my_s += fabs (xx (k)) ;
                        }
                    }
                    else
                    {
                        for (int64_t k = k1 ; k < k2 ; k++)
                        {
                            my_s += fabs (xx (k) - yy (k)) ;
                        }
                    }
                }
                break ;

                case INT64_MAX:     // inf-norm: max (abs (x-y))
                {
                    if (y == NULL)
                    {
                        for (int64_t k = k1 ; k < k2 ; k++)
                        {
                            my_s = fmax (my_s, fabs (xx (k))) ;
                        }
                    }
                    else
                    {
                        for (int64_t k = k1 ; k < k2 ; k++)
                        {
                            my_s = fmax (my_s, fabs (xx (k) - yy (k))) ;
                        }
                    }
                }
                break ;

                case INT64_MIN:     // (-inf)-norm: min (abs (x-y))
                {
                    my_s = INFINITY ;
                    if (y == NULL)
                    {
                        for (int64_t k = k1 ; k < k2 ; k++)
                        {
                            my_s = fmin (my_s, fabs (xx (k))) ;
                        }
                    }
                    else
                    {
                        for (int64_t k = k1 ; k < k2 ; k++)
                        {
                            my_s = fmin (my_s, fabs (xx (k) - yy (k))) ;
                        }
                    }
                }
                break ;

                default: ;  // p-norm not yet supported
            }

            Work [tid] = my_s ;
        }
    }

    //--------------------------------------------------------------------------
    // combine results of each thread
    //--------------------------------------------------------------------------

    double s = 0 ;
    switch (p)
    {
        case 0:     // Frobenius norm
        case 2:     // 2-norm: sqrt of sum of (x-y).^2
        {
            for (int64_t tid = 0 ; tid < nthreads ; tid++)
            {
                s += Work [tid] ;
            }
            s = sqrt (s) ;
        }
        break ;

        case 1:     // 1-norm: sum (abs (x-y))
        {
            for (int64_t tid = 0 ; tid < nthreads ; tid++)
            {
                s += Work [tid] ;
            }
        }
        break ;

        case INT64_MAX:     // inf-norm: max (abs (x-y))
        {
            for (int64_t tid = 0 ; tid < nthreads ; tid++)
            {
                s = fmax (s, Work [tid]) ;
            }
        }
        break ;

        case INT64_MIN:     // (-inf)-norm: min (abs (x-y))
        {
            s = Work [0] ;
            for (int64_t tid = 1 ; tid < nthreads ; tid++)
            {
                s = fmin (s, Work [tid]) ;
            }
        }
        break ;

        default:    // p-norm not yet supported
            s = -1 ;
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    GB_FREE_WORKSPACE ;
    return (s) ;
}

//------------------------------------------------------------------------------
// GB_make_shallow.c: force a matrix to have purely shallow components
//------------------------------------------------------------------------------

void GB_make_shallow (GrB_Matrix A)
{
    if (A == NULL) return ;
    A->p_shallow = (A->p != NULL) ;
    A->h_shallow = (A->h != NULL) ;
    A->b_shallow = (A->b != NULL) ;
    A->i_shallow = (A->i != NULL) ;
    A->x_shallow = (A->x != NULL) ;
    #ifdef GB_MEMDUMP
    printf ("remove from memtable: Ap:%p Ah:%p Ab:%p Ai:%p Ax:%p\n", // MEMDUMP
        A->p, A->h, A->b, A->i, A->x) ;
    #endif
    if (A->p != NULL) GB_Global_memtable_remove (A->p) ;
    if (A->h != NULL) GB_Global_memtable_remove (A->h) ;
    if (A->b != NULL) GB_Global_memtable_remove (A->b) ;
    if (A->i != NULL) GB_Global_memtable_remove (A->i) ;
    if (A->x != NULL) GB_Global_memtable_remove (A->x) ;
    GB_make_shallow (A->Y) ;
}

