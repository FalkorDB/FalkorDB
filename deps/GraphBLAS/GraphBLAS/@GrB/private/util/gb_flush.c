//------------------------------------------------------------------------------
// gb_flush: flush mexPrintf output to Command Window
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// When GraphBLAS is under development, it is essential that output from
// mexPrintf be flushed to the Command Window as soon as possible.  This
// function ensures that this is done.  The function is also used in
// production, when a built-in function wishes to display the contents of a GrB
// object, via disp or display.  This ensures that all output from GraphBLAS is
// immediately visible in the Command Window.

// gb_flush ( ) returns 0 if successful, or any nonzero value on failure.  This
// matches the behavior of the ANSI C fflush.

#include "gb_interface.h"

int gb_flush ( void )       // flush mexPrintf output to Command Window
{
    // 'drawnow' is slow when logging in remotely: disable it.
    // return (mexEvalString ("drawnow ; pause (1e-8) ;")) ;
    return (mexEvalString ("pause (1e-8) ;")) ;
}

