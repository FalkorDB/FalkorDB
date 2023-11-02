//------------------------------------------------------------------------------
// GB_mx_xsame64: check if two FP64 arrays are equal (ignoring zombies)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"

bool GB_mx_xsame64  // true if arrays X and Y are the same (ignoring zombies)
(
    double *X,  bool X_iso,
    double *Y,  bool Y_iso,
    int8_t *Xb,     // bitmap of X and Y (NULL if no bitmap)
    int64_t len,    // length of X and Y
    int64_t *I,     // row indices (for zombies), same length as X and Y
    double eps      // error tolerance allowed (eps > 0)
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
            double xi = GBX (X, i, X_iso) ;
            double yi = GBX (Y, i, Y_iso) ;
            int c = fpclassify (xi) ;
            if (c != fpclassify (yi)) return (false) ;
            if (c == FP_ZERO)
            {
                // both are zero, which is OK
            }
            else if (c == FP_INFINITE)
            {
                // + or -infinity
                if (xi != yi) return (false) ;
            }
            else if (c != FP_NAN)
            {
                // both are normal or subnormal, and nonzero
                double err = fabs (xi - yi) / fabs (xi) ;
                if (err > eps) return (false) ;
            }
        }
    }
    return (true) ;
}

