//------------------------------------------------------------------------------
// GB_helper.c: helper functions for @GrB interface
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// These functions are only used by the @GrB interface for
// SuiteSparse:GraphBLAS.

#include "helper/GB_helper.h"

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
    work_type *Work = GB_MALLOC_MEMORY (nthreads, sizeof (work_type),       \
        &Work_size) ;                                                       \
    if (Work == NULL) return (false) ;

//------------------------------------------------------------------------------
// GB_FREE_WORKSPACE: free per-thread workspace
//------------------------------------------------------------------------------

#define GB_FREE_WORKSPACE                                                   \
    GB_FREE_MEMORY (&Work, Work_size) ;

//------------------------------------------------------------------------------
// GB_helper5: construct pattern of S for gblogassign
//------------------------------------------------------------------------------

void GB_helper5             // construct pattern of S
(
    // output:
    uint64_t *restrict Si,          // array of size anz
    uint64_t *restrict Sj,          // array of size anz
    // input:
    const void *Mi,                 // array of size mnz, M->i, may be NULL
    const bool Mi_is_32,            // if true, M->i is 32-bit; else 64-bit
    const uint64_t *restrict Mj,    // array of size mnz
    const int64_t mvlen,            // M->vlen
    const void *Ai,                 // array of size anz, A->i, may be NULL
    const bool Ai_is_32,            // if true, A->i is 32-bit; else 64-bit
    const int64_t avlen,            // A->vlen
    const uint64_t anz
)
{

    GB_NTHREADS_HELPER (anz) ;
    ASSERT (Mj != NULL) ;
    ASSERT (Si != NULL) ;
    ASSERT (Sj != NULL) ;

    GB_IDECL (Ai, const, u) ; GB_IPTR (Ai, Ai_is_32) ;
    GB_IDECL (Mi, const, u) ; GB_IPTR (Mi, Mi_is_32) ;

    int64_t k ;
    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (k = 0 ; k < anz ; k++)
    {
        int64_t i = GBi_A (Ai, k, avlen) ;
        Si [k] = GBi_M (Mi, i, mvlen) ;
        Sj [k] = Mj [i] ;
    }
}

//------------------------------------------------------------------------------
// GB_helper7: Kx = uint64 (0:mnz-1), for gblogextract
//------------------------------------------------------------------------------

// TODO: use GrB_apply with a positional operator instead

void GB_helper7              // Kx = uint64 (0:mnz-1)
(
    uint64_t *restrict Kx,           // array of size mnz
    const uint64_t mnz
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
    uint64_t n
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
// persistent Container
//------------------------------------------------------------------------------

static GxB_Container Container = NULL ;

static GrB_Vector GB_helper_component (void)
{
    size_t s = sizeof (struct GB_Vector_opaque) ;
    GrB_Vector p = GB_Global_persistent_malloc (s) ;
    if (p != NULL)
    {
        memset (p, 0, s) ;
        p->header_size = s ;
        p->type = GrB_BOOL ;
        p->is_csc = true ;
        p->plen = -1 ;
        p->vdim = 1 ;
        p->nvec = 1 ;
        p->sparsity_control = GxB_FULL ;
        p->magic = GB_MAGIC ;
    }
    ASSERT_VECTOR_OK (p, "container component", GB0) ;
    return (p) ;
}

GxB_Container GB_helper_container (void)    // return the global Container
{
    return (Container) ;
}

void GB_helper_container_new (void)         // allocate the global Container
{
    // free any existing Container
    GB_helper_container_free ( ) ;

    // allocate a new Container
    size_t s = sizeof (struct GxB_Container_struct) ;
    Container = GB_Global_persistent_malloc (s) ;
    if (Container != NULL)
    {
        memset (Container, 0, s) ;
        Container->p = GB_helper_component ( ) ;
        Container->h = GB_helper_component ( ) ;
        Container->b = GB_helper_component ( ) ;
        Container->i = GB_helper_component ( ) ;
        Container->x = GB_helper_component ( ) ;

        // clear the Container scalars
        Container->nrows_nonempty = -1 ;
        Container->ncols_nonempty = -1 ;
        Container->format = GxB_FULL ;
        Container->orientation = GrB_ROWMAJOR ;
    }
}

void GB_helper_container_free (void)        // free the global Container
{
    if (Container == NULL) return ;
    GB_Global_persistent_free ((void **) &(Container->p)) ;
    GB_Global_persistent_free ((void **) &(Container->h)) ;
    GB_Global_persistent_free ((void **) &(Container->b)) ;
    GB_Global_persistent_free ((void **) &(Container->i)) ;
    GB_Global_persistent_free ((void **) &(Container->x)) ;
    GB_Global_persistent_free ((void **) &(Container)) ;
}

