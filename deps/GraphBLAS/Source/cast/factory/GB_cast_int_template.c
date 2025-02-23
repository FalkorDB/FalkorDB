//------------------------------------------------------------------------------
// GB_cast_int_template: parallel int32_t/int64_t/uint32_t/uint64_t type cast
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

{
    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (k = 0 ; k < n ; k++)
    {
        Dest [k] = Src [k] ;
    }
}

