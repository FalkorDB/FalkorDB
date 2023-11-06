//------------------------------------------------------------------------------
// GB_bld_template.c: Tx=build(Sx), and assemble any duplicate tuples
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This template is used in GB_builder and the FactoryKernels/GB_bld__* workers.
// This is the same for both vectors and matrices, since this step is agnostic
// about which vectors the entries appear.

// Sx and Tx are either both iso or both non-iso.  For the iso case,
// GB_ISO_BUILD is defined, and K_work is NULL.  The iso case is not handled by
// the FactoryKernels/GB_bld__* workers, since it doesn't access the values at all.

{

    // k unused for some uses of this template
    #include "GB_unused.h"

    if (ndupl == 0)
    {

        //----------------------------------------------------------------------
        // no duplicates, just permute Sx into Tx
        //----------------------------------------------------------------------

        // If no duplicates are present, then GB_builder has already
        // transplanted I_work into T->i, so this step does not need to
        // construct T->i.  The tuple values, in Sx, are copied or permuted
        // into T->x.  This step is skipped if T and Sx are iso.

        #ifndef GB_ISO_BUILD

            if (K_work == NULL)
            {

                int tid ;
                #pragma omp parallel for num_threads(nthreads) schedule(static)
                for (tid = 0 ; tid < nthreads ; tid++)
                {
                    int64_t tstart = tstart_slice [tid] ;
                    int64_t tend   = tstart_slice [tid+1] ;
                    for (int64_t t = tstart ; t < tend ; t++)
                    { 
                        // Tx [t] = (ttype) Sx [t] ;
                        GB_BLD_COPY (Tx, t, Sx, t) ;
                    }
                }

            }
            else
            {

                int tid ;
                #pragma omp parallel for num_threads(nthreads) schedule(static)
                for (tid = 0 ; tid < nthreads ; tid++)
                {
                    int64_t tstart = tstart_slice [tid] ;
                    int64_t tend   = tstart_slice [tid+1] ;
                    for (int64_t t = tstart ; t < tend ; t++)
                    { 
                        // Tx [t] = (ttype) Sx [K_work [t]] ;
                        GB_BLD_COPY (Tx, t, Sx, K_work [t]) ;
                    }
                }
            }

        #endif

    }
    else
    {

        //----------------------------------------------------------------------
        // assemble duplicates
        //----------------------------------------------------------------------

        // If T and Sx as non-iso, entries in Sx must be copied into T->x, with
        // any duplicates summed via the operator.  T->i must also be
        // constructed.  T->x and Sx are not modified if they are iso.

        int tid ;
        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (tid = 0 ; tid < nthreads ; tid++)
        {
            int64_t my_tnz = tnz_slice [tid] ;
            int64_t tstart = tstart_slice [tid] ;
            int64_t tend   = tstart_slice [tid+1] ;

            // find the first unique tuple owned by this slice
            int64_t t ;
            for (t = tstart ; t < tend ; t++)
            { 
                // get the tuple and break if it is not a duplicate
                if (I_work [t] >= 0) break ;
            }

            // scan all tuples and assemble any duplicates
            for ( ; t < tend ; t++)
            {
                // get the t-th tuple, a unique tuple
                int64_t i = I_work [t] ;
                ASSERT (i >= 0) ;
                #ifndef GB_ISO_BUILD
                int64_t k = (K_work == NULL) ? t : K_work [t] ;
                // Tx [my_tnz] = (ttype) Sx [k] ;
                GB_BLD_COPY (Tx, my_tnz, Sx, k) ;
                #endif
                Ti [my_tnz] = i ;

                // assemble all duplicates that follow it.  This may assemble
                // the first duplicates in the next slice(s) (up to but not
                // including the first unique tuple in the subsequent slice(s)).
                for ( ; t+1 < nvals && I_work [t+1] < 0 ; t++)
                { 
                    // assemble the duplicate tuple
                    #ifndef GB_ISO_BUILD
                    int64_t k = (K_work == NULL) ? (t+1) : K_work [t+1] ;
                    // Tx [my_tnz] += Sx [k], typecasting as needed
                    GB_BLD_DUP (Tx, my_tnz, Sx, k) ;
                    #endif
                }
                my_tnz++ ;
            }
        }
    }
}

#undef GB_ISO_BUILD

