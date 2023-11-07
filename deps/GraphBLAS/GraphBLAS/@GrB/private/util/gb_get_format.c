//------------------------------------------------------------------------------
// gb_get_format: determine the format of a matrix result 
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// gb_get_format determines the format of a result matrix C, which may be
// computed from one or two input matrices A and B.  The following rules are
// used, in order:

// (1) GraphBLAS operations of the form C = GrB.method (Cin, ...) use the
//      format of Cin for the new matrix C.

// (1) If the format is determined by the descriptor to the method, then that
//      determines the format of C.

// (2) If C is a column vector (cncols == 1) then C is stored by column.

// (3) If C is a row vector (cnrows == 1) then C is stored by row.

// (4) If A is present, and not a row or column vector or scalar, then its
//      format is used for C.

// (5) If B is present, and not a row or column vector or scalar, then its
//      format is used for C.

// (6) Otherwise, the global default format is used for C.

#include "gb_interface.h"

GxB_Format_Value gb_get_format          // GxB_BY_ROW or GxB_BY_COL
(
    GrB_Index cnrows,                   // C is cnrows-by-cncols
    GrB_Index cncols,
    GrB_Matrix A,                       // may be NULL
    GrB_Matrix B,                       // may be NULL
    GxB_Format_Value fmt_descriptor     // may be GxB_NO_FORMAT
)
{

    GxB_Format_Value fmt ;

    if (fmt_descriptor != GxB_NO_FORMAT)
    { 
        // (1) the format is defined by the descriptor to the method
        fmt = fmt_descriptor ;
    }
    else if (cncols == 1)
    { 
        // (2) column vectors are stored by column, by default
        fmt = GxB_BY_COL ;
    }
    else if (cnrows == 1)
    { 
        // (3) row vectors are stored by row, by default
        fmt = GxB_BY_ROW ;
    }
    else if (A != NULL && !gb_is_vector (A))
    { 
        // (4) get the format of A
        OK (GxB_Matrix_Option_get (A, GxB_FORMAT, &fmt)) ;
    }
    else if (B != NULL && !gb_is_vector (B))
    { 
        // (5) get the format of B
        OK (GxB_Matrix_Option_get (B, GxB_FORMAT, &fmt)) ;
    }
    else
    { 
        // (6) get the global default format
        OK (GxB_Global_Option_get (GxB_FORMAT, &fmt)) ;
    }
    return (fmt) ;
}

