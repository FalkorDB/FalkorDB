//------------------------------------------------------------------------------
// GB_mex_test18: demacrofy tests
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"
#include "GB_mex_errors.h"
#include "GB_stringify.h"

#define USAGE "GB_mex_test18"

#define FREE_ALL ;
#define GET_DEEP_COPY ;
#define FREE_DEEP_COPY ;

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    //--------------------------------------------------------------------------
    // startup GraphBLAS
    //--------------------------------------------------------------------------

    GrB_Info info ;
    bool malloc_debug = GB_mx_get_global (true) ;

    //--------------------------------------------------------------------------
    // test GB_demacrofy_name
    //--------------------------------------------------------------------------

    uint64_t scode ;
    char *name_space, *kname, *suffix ;
    char name [4096] ;

    strcpy (name, "bad") ;
    info = GB_demacrofy_name (name, &name_space, &kname, &scode, &suffix) ;
    CHECK (info == GrB_NO_VALUE) ;

    strcpy (name, "alsobad") ;
    info = GB_demacrofy_name (name, &name_space, &kname, &scode, &suffix) ;
    CHECK (info == GrB_NO_VALUE) ;

    strcpy (name, "space__kname__007__suffix") ;
    info = GB_demacrofy_name (name, &name_space, &kname, &scode, &suffix) ;
    CHECK (info == GrB_SUCCESS) ;
    CHECK (MATCH (name_space, "space")) ;
    CHECK (MATCH (kname, "kname")) ;
    CHECK (MATCH (suffix, "suffix")) ;
    CHECK (scode == 7) ;

    strcpy (name, "space__kname__mangle__suffix") ;
    info = GB_demacrofy_name (name, &name_space, &kname, &scode, &suffix) ;
    CHECK (info == GrB_NO_VALUE) ;

    strcpy (name, "morespace__morekname__042") ;
    info = GB_demacrofy_name (name, &name_space, &kname, &scode, &suffix) ;
    CHECK (info == GrB_SUCCESS) ;
    CHECK (MATCH (name_space, "morespace")) ;
    CHECK (MATCH (kname, "morekname")) ;
    CHECK (suffix == NULL) ;
    CHECK (scode == 0x42) ;

    //--------------------------------------------------------------------------
    // finalize GraphBLAS
    //--------------------------------------------------------------------------

    GB_mx_put_global (true) ;
    printf ("\nGB_mex_test18:  all tests passed\n\n") ;
}

