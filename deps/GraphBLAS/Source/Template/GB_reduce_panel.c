//------------------------------------------------------------------------------
// GB_reduce_panel: z=reduce(A), reduce a matrix to a scalar
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Reduce a matrix to a scalar using a panel-based method for built-in
// operators or jitified kernels.  A must be sparse, hypersparse, or full (it
// cannot be bitmap).  A cannot have any zombies.  If A has zombies or is
// bitmap, GB_reduce_to_scalar_template is used instead.  A is not iso.

// A generic method via memcpy is not supported.

// The Panel and W arrays always have the same type as z, GB_Z_TYPE.

#include "GB_unused.h"

// default panel size
#ifndef GB_PANEL
#define GB_PANEL 16
#endif

{

    //--------------------------------------------------------------------------
    // get A
    //--------------------------------------------------------------------------

    const GB_A_TYPE *restrict Ax = (GB_A_TYPE *) A->x ;
    ASSERT (!A->iso) ;
    GB_A_NVALS (anz) ;      // int64_t anz = GB_nnz (A) ;
    ASSERT (anz > 0) ;
    ASSERT (!GB_IS_BITMAP (A)) ;
    ASSERT (A->nzombies == 0) ;
    GB_DECLARE_TERMINAL_CONST (zterminal) ;

    #if GB_IS_ANY_MONOID
    // the ANY monoid can take any entry, and terminate immediately
    // z = (ztype) Ax [anz-1]
    GB_GETA (z, Ax, anz-1, false) ;
    #else

    //--------------------------------------------------------------------------
    // reduce A to a scalar
    //--------------------------------------------------------------------------

    if (nthreads == 1)
    {

        //----------------------------------------------------------------------
        // load the Panel with the first entries
        //----------------------------------------------------------------------

        GB_Z_TYPE Panel [GB_PANEL] ;
        int64_t first_panel_size = GB_IMIN (GB_PANEL, anz) ;
        for (int64_t k = 0 ; k < first_panel_size ; k++)
        { 
            // Panel [k] = (ztype) Ax [k] ;
            GB_GETA (Panel [k], Ax, k, false) ;
        }

        #if GB_MONOID_IS_TERMINAL
        int panel_count = 0 ;
        #endif

        //----------------------------------------------------------------------
        // reduce all entries to the Panel
        //----------------------------------------------------------------------

        for (int64_t p = GB_PANEL ; p < anz ; p += GB_PANEL)
        {
            if (p + GB_PANEL > anz)
            {
                // last partial panel
                for (int64_t k = 0 ; k < anz-p ; k++)
                { 
                    // Panel [k] += (ztype) Ax [p+k]
                    GB_GETA_AND_UPDATE (Panel [k], Ax, p+k) ;
                }
            }
            else
            {
                // whole panel
                for (int64_t k = 0 ; k < GB_PANEL ; k++)
                { 
                    // Panel [k] += (ztype) Ax [p+k]
                    GB_GETA_AND_UPDATE (Panel [k], Ax, p+k) ;
                }
                #if GB_MONOID_IS_TERMINAL
                panel_count-- ;
                if (panel_count <= 0)
                {
                    // check for early exit only every 256 panels
                    panel_count = 256 ;
                    int count = 0 ;
                    for (int64_t k = 0 ; k < GB_PANEL ; k++)
                    { 
                        count += GB_TERMINAL_CONDITION (Panel [k], zterminal) ;
                    }
                    if (count > 0)
                    { 
                        break ;
                    }
                }
                #endif
            }
        }

        //----------------------------------------------------------------------
        // z = reduce (Panel)
        //----------------------------------------------------------------------

        z = Panel [0] ;
        for (int64_t k = 1 ; k < first_panel_size ; k++)
        { 
            // z += Panel [k]
            GB_UPDATE (z, Panel [k]) ;
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // all tasks share a single early_exit flag
        //----------------------------------------------------------------------

        // If this flag gets set, all tasks can terminate early

        #if GB_MONOID_IS_TERMINAL
        bool early_exit = false ;
        #endif

        //----------------------------------------------------------------------
        // each thread reduces its own slice in parallel
        //----------------------------------------------------------------------

        int tid ;
        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (tid = 0 ; tid < ntasks ; tid++)
        {

            //------------------------------------------------------------------
            // determine the work for this task
            //------------------------------------------------------------------

            // Task tid reduces Ax [pstart:pend-1] to the scalar W [tid]

            int64_t pstart, pend ;
            GB_PARTITION (pstart, pend, anz, tid, ntasks) ;
            GB_Z_TYPE t ;
            // t = (ztype) Ax [pstart]
            GB_GETA (t, Ax, pstart, false) ;

            //------------------------------------------------------------------
            // skip this task if the terminal value has already been reached
            //------------------------------------------------------------------

            #if GB_MONOID_IS_TERMINAL
            // check if another task has called for an early exit
            bool my_exit ;

            GB_ATOMIC_READ
            my_exit = early_exit ;

            if (!my_exit)
            #endif

            //------------------------------------------------------------------
            // do the reductions for this task
            //------------------------------------------------------------------

            {

                //--------------------------------------------------------------
                // load the Panel with the first entries
                //--------------------------------------------------------------

                GB_Z_TYPE Panel [GB_PANEL] ;
                int64_t my_anz = pend - pstart ;
                int64_t first_panel_size = GB_IMIN (GB_PANEL, my_anz) ;
                for (int64_t k = 0 ; k < first_panel_size ; k++)
                { 
                    // Panel [k] = (ztype) Ax [pstart + k] ;
                    GB_GETA (Panel [k], Ax, pstart + k, false) ;
                }

                #if GB_MONOID_IS_TERMINAL
                int panel_count = 0 ;
                #endif

                //--------------------------------------------------------------
                // reduce all entries to the Panel
                //--------------------------------------------------------------

                for (int64_t p = pstart + GB_PANEL ; p < pend ; p += GB_PANEL)
                {
                    if (p + GB_PANEL > pend)
                    {
                        // last partial panel
                        for (int64_t k = 0 ; k < pend-p ; k++)
                        { 
                            // Panel [k] += (ztype) Ax [p+k]
                            GB_GETA_AND_UPDATE (Panel [k], Ax, p+k) ;
                        }
                    }
                    else
                    {
                        // whole panel
                        for (int64_t k = 0 ; k < GB_PANEL ; k++)
                        { 
                            // Panel [k] += (ztype) Ax [p+k]
                            GB_GETA_AND_UPDATE (Panel [k], Ax, p+k) ;
                        }
                        #if GB_MONOID_IS_TERMINAL
                        panel_count-- ;
                        if (panel_count <= 0)
                        {
                            // check for early exit only every 256 panels
                            panel_count = 256 ;
                            int count = 0 ;
                            for (int64_t k = 0 ; k < GB_PANEL ; k++)
                            { 
                                count += GB_TERMINAL_CONDITION (Panel [k],
                                    zterminal) ;
                            }
                            if (count > 0)
                            { 
                                break ;
                            }
                        }
                        #endif
                    }
                }

                //--------------------------------------------------------------
                // t = reduce (Panel)
                //--------------------------------------------------------------

                t = Panel [0] ;
                for (int64_t k = 1 ; k < first_panel_size ; k++)
                { 
                    // t += Panel [k]
                    GB_UPDATE (t, Panel [k]) ;
                }

                #if GB_MONOID_IS_TERMINAL
                if (GB_TERMINAL_CONDITION (t, zterminal))
                { 
                    // tell all other tasks to exit early
                    GB_ATOMIC_WRITE
                    early_exit = true ;
                }
                #endif
            }

            //------------------------------------------------------------------
            // save the results of this task
            //------------------------------------------------------------------

            W [tid] = t ;
        }

        //----------------------------------------------------------------------
        // sum up the results of each slice using a single thread
        //----------------------------------------------------------------------

        z = W [0] ;
        for (int tid = 1 ; tid < ntasks ; tid++)
        { 
            // z += W [tid]
            GB_UPDATE (z, W [tid]) ;
        }
    }
    #endif
}

