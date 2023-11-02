//------------------------------------------------------------------------------
// GB_mx_xsame: check if two arrays are equal (ignoring zombies)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"

bool GB_mx_xsame    // true if arrays X and Y are the same (ignoring zombies)
(
    char *X,    bool X_iso,
    char *Y,    bool Y_iso,
    int8_t *Xb,     // bitmap of X and Y (NULL if no bitmap)
    int64_t len,    // length of X and Y
    size_t s,       // size of each entry of X and Y
    int64_t *I      // row indices (for zombies), same length as X and Y
)
{
    if (X == Y) return (true) ;
    if (X == NULL) return (false) ;
    if (Y == NULL) return (false) ;
    for (int64_t i = 0 ; i < len ; i++)
    {
        if (Xb != NULL && Xb [i] == 0)
        {
            // ignore X [i] and Y [i] if they are not in the bitmap
            continue ;
        }
        // check X [i] and Y [i], but ignore zombies
        if (I == NULL || I [i] >= 0)
        {
            if (!GB_mx_same (X+(X_iso ? 0:(i*s)), Y+(Y_iso ? 0:(i*s)), s))
            {
                return (false) ;
            }
        }
    }
    return (true) ;
}

