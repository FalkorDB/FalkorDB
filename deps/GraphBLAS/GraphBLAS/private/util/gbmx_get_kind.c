//------------------------------------------------------------------------------
// gbmx_get_kind: get descriptor.kind from a MATLAB descriptor struct
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

kind_enum_t gbmx_get_kind
(
    const mxArray *mxdesc
)
{

    kind_enum_t kind = KIND_GRB ;

    if (mxdesc != NULL && mxIsStruct (mxdesc))
    {
        mxArray *mxkind = mxGetField (mxdesc, 0, "kind") ;
        if (mxkind != NULL)
        {
            // get the string from the built-in field
            char s [LEN+2] ;
            gbmx_mxstring_to_string (s, LEN, mxkind, "kind") ;
            if (MATCH (s, "grb") || MATCH (s, "default") || MATCH (s, "ghb"))
            { 
                // both gbdesc.kind == KIND_GHB and gbdesc.kind = KIND_GRB
                // selects GrB for a GrB.method, and GhB for a GhB.method.
                kind = KIND_GRB ;           // GrB or GhB matrix
            }
            else if (MATCH (s, "sparse"))
            { 
                kind = KIND_SPARSE ;        // built-in sparse matrix
            }
            else if (MATCH (s, "full"))
            { 
                kind = KIND_FULL ;          // built-in full matrix
            }
            else if (MATCH (s, "builtin")   // preferred
                || MATCH (s, "matlab")      // deprecated (use 'builtin')
                || MATCH (s, "octave"))     // 'builtin' is preferred
            { 
                kind = KIND_BUILTIN ;       // built-in sparse or full matrix
            }
            else
            { 
                ERROR ("invalid descriptor.kind", GrB_INVALID_VALUE) ;
            }
        }
    }

    return (kind) ;
}

