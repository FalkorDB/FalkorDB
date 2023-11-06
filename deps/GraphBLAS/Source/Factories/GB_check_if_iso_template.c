//------------------------------------------------------------------------------
// GB_check_if_iso_template: check if all entries in a matrix are identical
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

{

    //--------------------------------------------------------------------------
    // get A
    //--------------------------------------------------------------------------

    const GB_A_TYPE *restrict Ax = (GB_A_TYPE *) A->x ;

    //--------------------------------------------------------------------------
    // check all entries to see if they are equal to the first entry
    //--------------------------------------------------------------------------

    int tid ;
    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
    for (tid = 0 ; tid < ntasks ; tid++)
    {
        int64_t pstart, pend ;
        GB_PARTITION (pstart, pend, anz, tid, ntasks) ;
        bool my_iso ;
        GB_ATOMIC_READ
        my_iso = iso ;
        if (my_iso)
        {
            // GB_A_TYPE a = Ax [0] ;
            GB_GET_FIRST_VALUE (GB_A_TYPE, a, Ax) ;
            for (int64_t p = pstart ; my_iso && p < pend ; p++)
            { 
                // my_iso = my_iso && (a == Ax [p])
                GB_COMPARE_WITH_FIRST_VALUE (my_iso, a, Ax, p) ;
            }
            if (!my_iso)
            { 
                // tell the other tasks to exit early
                GB_ATOMIC_WRITE
                iso = false ;
            }
        }
    }
    done = true ;
}

#undef GB_A_TYPE

