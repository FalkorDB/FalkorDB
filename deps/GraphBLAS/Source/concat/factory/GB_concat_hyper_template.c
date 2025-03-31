//------------------------------------------------------------------------------
// GB_concat_hyper_template: create tuples from one tile
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// WORK_I and WORK_J are macro arguments, either uint32_t or uint64_t arrays.

{
    // if A is iso but C is not, extractTuples expands A->x [0] into
    // all Wx [...].   If both A and C are iso, then all tiles are iso,
    // and Wx is not extracted.

    // FUTURE: could revise GB_extractTuples to take in an offset instead

    if (csc)
    {
        GB_OK (GB_extractTuples (WORK_I + pC, Ci_is_32, WORK_J + pC, Cj_is_32,
            (C_iso) ? NULL : (Wx + pC * csize),
            (uint64_t *) (&anz), ctype, A, Werk)) ;
    }
    else
    {
        GB_OK (GB_extractTuples (WORK_J + pC, Cj_is_32, WORK_I + pC, Ci_is_32,
            (C_iso) ? NULL : (Wx + pC * csize),
            (uint64_t *) (&anz), ctype, A, Werk)) ;
    }

    // adjust the indices to reflect their new place in C

    if (cistart > 0 && cvstart > 0)
    { 
        #pragma omp parallel for num_threads(nth) schedule(static)
        for (pA = 0 ; pA < anz ; pA++)
        {
            WORK_I [pC + pA] += cistart ;
            WORK_J [pC + pA] += cvstart ;
        }
    }
    else if (cistart > 0)
    { 
        #pragma omp parallel for num_threads(nth) schedule(static)
        for (pA = 0 ; pA < anz ; pA++)
        {
            WORK_I [pC + pA] += cistart ;
        }
    }
    else if (cvstart > 0)
    { 
        #pragma omp parallel for num_threads(nth) schedule(static)
        for (pA = 0 ; pA < anz ; pA++)
        {
            WORK_J [pC + pA] += cvstart ;
        }
    }
}

#undef WORK_I
#undef WORK_J

