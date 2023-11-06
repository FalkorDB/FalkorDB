//------------------------------------------------------------------------------
// gb_find_dot:  find the first two occurences of '.' in a string
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// s has the form 'plus.times.double' for a semiring, or 'plus.single' for a
// binary operator, or 'abs.int8' for a unary operator.  In all such cases, one
// or two dots ('.') must be found.  If no dots are found, position [0] and [1]
// are both -1.  If one dot is found, s [position [0]] == '.', and position [1]
// is -1.  If two dots are found, s [position [0]] == '.' is the first dot, and
// s [position [1]] == '.' is the second.

#include "gb_interface.h"

void gb_find_dot            // find 1st and 2nd dot ('.') in a string
(
    int32_t position [2],   // positions of one or two dots
    const char *s           // null-terminated string to search
)
{

    position [0] = -1 ;
    position [1] = -1 ;

    for (int32_t k = 0 ; s [k] != '\0' ; k++)
    {
        if (s [k] == '.')
        {
            if (position [0] == -1)
            { 
                // first dot has been found
                position [0] = k ;
            }
            else
            { 
                // 2nd dot has been found; the search is done
                position [1] = k ;
                return ;
            }
        }
    }
}

