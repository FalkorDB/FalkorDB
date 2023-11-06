//------------------------------------------------------------------------------
// GB_unjumbled_template.c: unjumble the vectors of a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

{
    int tid ;
    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
    for (tid = 0 ; tid < ntasks ; tid++)
    {

        //----------------------------------------------------------------------
        // get the task description
        //----------------------------------------------------------------------

        const int64_t kfirst = A_slice [tid] ;
        const int64_t klast  = A_slice [tid+1] ;

        //----------------------------------------------------------------------
        // sort vectors kfirst to klast
        //----------------------------------------------------------------------

        for (int64_t k = kfirst ; k < klast ; k++)
        {

            //------------------------------------------------------------------
            // check if the vector needs sorting
            //------------------------------------------------------------------

            bool jumbled = false ;
            const int64_t pA_start = Ap [k] ;
            const int64_t pA_end   = Ap [k+1] ;
            int64_t ilast = -1 ;
            for (int64_t pA = pA_start ; pA < pA_end ; pA++)
            {
                int64_t i = Ai [pA] ;
                if (i < ilast)
                { 
                    jumbled = true ;
                    break ;
                }
                ilast = i ;
            }

            //------------------------------------------------------------------
            // sort the vector
            //------------------------------------------------------------------

            if (jumbled)
            { 
                const int64_t aknz = pA_end - pA_start ;
                GB_QSORT ;
            }
        }
    }
}

#undef GB_QSORT

