//------------------------------------------------------------------------------
// GB_reduce_to_scalar_template: z=reduce(A), reduce a matrix to a scalar
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Reduce a matrix to a scalar, with typecasting and generic operators.
// No panel is used.  The workspace W always has the same type as the ztype
// of the monoid, GB_Z_TYPE.

#include "GB_unused.h"

// z += W [i], no typecast
#ifndef GB_ADD_ARRAY_TO_SCALAR
#define GB_ADD_ARRAY_TO_SCALAR(z,W,i) GB_UPDATE (z, W [i])
#endif

// W [k] = z, no typecast
#ifndef GB_COPY_SCALAR_TO_ARRAY
#define GB_COPY_SCALAR_TO_ARRAY(W,k,z) W [k] = z
#endif

{

    //--------------------------------------------------------------------------
    // get A
    //--------------------------------------------------------------------------

    const int8_t   *restrict Ab = A->b ;
    const int64_t  *restrict Ai = A->i ;
    const GB_A_TYPE *restrict Ax = (GB_A_TYPE *) A->x ;
    GB_A_NHELD (anz) ;      // int64_t anz = GB_nnz_held (A) ;
    ASSERT (anz > 0) ;
    #ifdef GB_JIT_KERNEL
    #define A_has_zombies GB_A_HAS_ZOMBIES
    #else
    const bool A_has_zombies = (A->nzombies > 0) ;
    #endif
    ASSERT (!A->iso) ;
    GB_DECLARE_TERMINAL_CONST (zterminal) ;

    //--------------------------------------------------------------------------
    // reduce A to a scalar
    //--------------------------------------------------------------------------

    if (nthreads == 1)
    {

        //----------------------------------------------------------------------
        // single thread
        //----------------------------------------------------------------------

        for (int64_t p = 0 ; p < anz ; p++)
        { 
            // skip if the entry is a zombie or if not in the bitmap
            if (A_has_zombies && GB_IS_ZOMBIE (Ai [p])) continue ;
            if (!GBB_A (Ab, p)) continue ;
            // z += (ztype) Ax [p]
            GB_GETA_AND_UPDATE (z, Ax, p) ;
            #if GB_MONOID_IS_TERMINAL
            // check for early exit
            GB_IF_TERMINAL_BREAK (z, zterminal) ;
            #endif
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // each thread reduces its own slice in parallel
        //----------------------------------------------------------------------

        bool early_exit = false ;
        int tid ;

        #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
        for (tid = 0 ; tid < ntasks ; tid++)
        {
            int64_t pstart, pend ;
            GB_PARTITION (pstart, pend, anz, tid, ntasks) ;
            // ztype t = identity
            GB_DECLARE_IDENTITY (t) ;
            bool my_exit, found = false ;
            GB_ATOMIC_READ
            my_exit = early_exit ;
            if (!my_exit)
            {
                for (int64_t p = pstart ; p < pend ; p++)
                { 
                    // skip if the entry is a zombie or if not in the bitmap
                    if (A_has_zombies && GB_IS_ZOMBIE (Ai [p])) continue ;
                    if (!GBB_A (Ab, p)) continue ;
                    found = true ;
                    // t += (ztype) Ax [p]
                    GB_GETA_AND_UPDATE (t, Ax, p) ;
                    #if GB_MONOID_IS_TERMINAL
                    // check for early exit
                    if (GB_TERMINAL_CONDITION (t, zterminal))
                    { 
                        // tell the other tasks to exit early
                        GB_ATOMIC_WRITE
                        early_exit = true ;
                        break ;
                    }
                    #endif
                }
            }
            F [tid] = found ;
            // W [tid] = t, no typecast
            GB_COPY_SCALAR_TO_ARRAY (W, tid, t) ;
        }

        //----------------------------------------------------------------------
        // sum up the results of each slice using a single thread
        //----------------------------------------------------------------------

        for (int tid = 0 ; tid < ntasks ; tid++)
        {
            if (F [tid])
            { 
                // z += W [tid], no typecast
                GB_ADD_ARRAY_TO_SCALAR (z, W, tid) ;
            }
        }
    }
}

