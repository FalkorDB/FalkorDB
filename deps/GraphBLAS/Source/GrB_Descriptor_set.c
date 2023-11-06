//------------------------------------------------------------------------------
// GrB_Descriptor_set: set a field in a descriptor
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

GrB_Info GrB_Descriptor_set     // set a parameter in a descriptor
(
    GrB_Descriptor desc,        // descriptor to modify
    GrB_Desc_Field field,       // parameter to change
    GrB_Desc_Value value        // value to change it to
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (desc != NULL && desc->header_size == 0)
    { 
        // built-in descriptors may not be modified
        return (GrB_INVALID_VALUE) ;
    }

    GB_WHERE (desc, "GrB_Descriptor_set (desc, field, value)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (desc) ;
    ASSERT_DESCRIPTOR_OK (desc, "desc to set", GB0) ;

    //--------------------------------------------------------------------------
    // set the parameter
    //--------------------------------------------------------------------------

    switch (field)
    {

        case GrB_OUTP : 

            if (! (value == GxB_DEFAULT || value == GrB_REPLACE))
            { 
                GB_ERROR (GrB_INVALID_VALUE,
                    "invalid descriptor value [%d] for GrB_OUTP field;\n"
                    "must be GxB_DEFAULT [%d] or GrB_REPLACE [%d]",
                    (int) value, (int) GxB_DEFAULT, (int) GrB_REPLACE) ;
            }
            desc->out = value ;
            break ;

        case GrB_MASK : 
        {

            if (! (value == GxB_DEFAULT ||
                   value == GrB_COMP ||
                   value == GrB_STRUCTURE ||
                   value == (GrB_COMP + GrB_STRUCTURE)))
            { 
                GB_ERROR (GrB_INVALID_VALUE,
                    "invalid descriptor value [%d] for GrB_MASK field;\n"
                    "must be GxB_DEFAULT [%d], GrB_COMP [%d],\n"
                    "GrB_STRUCTURE [%d], or GrB_COMP+GrB_STRUCTURE [%d]",
                    (int) value, (int) GxB_DEFAULT, (int) GrB_COMP,
                    (int) GrB_STRUCTURE,
                    (int) (GrB_COMP + GrB_STRUCTURE)) ;
            }
            int mask = (int) desc->mask ;
            switch (value)
            {
                case GrB_COMP      : mask |= GrB_COMP ;      break ;
                case GrB_STRUCTURE : mask |= GrB_STRUCTURE ; break ;
                default            : mask = (int) value ;    break ;
            }
            desc->mask = (GrB_Desc_Value) mask ;
        }
        break ;

        case GrB_INP0 : 

            if (! (value == GxB_DEFAULT || value == GrB_TRAN))
            { 
                GB_ERROR (GrB_INVALID_VALUE,
                    "invalid descriptor value [%d] for GrB_INP0 field;\n"
                    "must be GxB_DEFAULT [%d] or GrB_TRAN [%d]",
                    (int) value, (int) GxB_DEFAULT, (int) GrB_TRAN) ;
            }
            desc->in0 = value ;
            break ;

        case GrB_INP1 : 

            if (! (value == GxB_DEFAULT || value == GrB_TRAN))
            { 
                GB_ERROR (GrB_INVALID_VALUE,
                    "invalid descriptor value [%d] for GrB_INP1 field;\n"
                    "must be GxB_DEFAULT [%d] or GrB_TRAN [%d]",
                    (int) value, (int) GxB_DEFAULT, (int) GrB_TRAN) ;
            }
            desc->in1 = value ;
            break ;

        case GxB_AxB_METHOD : 

            if (! (value == GxB_DEFAULT  || value == GxB_AxB_GUSTAVSON
                || value == GxB_AxB_DOT
                || value == GxB_AxB_HASH || value == GxB_AxB_SAXPY))
            { 
                GB_ERROR (GrB_INVALID_VALUE,
                    "invalid descriptor value [%d] for GrB_AxB_METHOD field;\n"
                    "must be GxB_DEFAULT [%d], GxB_AxB_GUSTAVSON [%d]\n"
                    "GxB_AxB_DOT [%d], GxB_AxB_HASH [%d] or GxB_AxB_SAXPY [%d]",
                    (int) value, (int) GxB_DEFAULT, (int) GxB_AxB_GUSTAVSON,
                    (int) GxB_AxB_DOT,
                    (int) GxB_AxB_HASH, (int) GxB_AxB_SAXPY) ;
            }
            desc->axb = value ;
            break ;

        case GxB_IMPORT : 

            // The user application might not check the error return value of
            // this method, so do not return an error if the value is something
            // other that GxB_FAST_IMPORT (equal to GxB_DEFAULT) or
            // GxB_SERCURE_IMPORT.  Instead, default to slower but secure
            // import/deserialization, if the GxB_IMPORT setting is made.
            // Only use the fast import/deserialize if the value is GxB_DEFAULT
            // or GxB_FAST_IMPORT; otherwise use the slower secure method.
            desc->import =
                (value == GxB_DEFAULT) ? GxB_FAST_IMPORT : GxB_SECURE_IMPORT ;
            break ;

        default : 

            GB_ERROR (GrB_INVALID_VALUE,
                "invalid descriptor field [%d], must be one of:\n"
                "GrB_OUTP [%d], GrB_MASK [%d], GrB_INP0 [%d], GrB_INP1 [%d], "
                "GxB_AxB_METHOD [%d] or GxB_IMPORT [%d] (use GxB_Desc_set "
                "for other descriptor settings)", (int) field, (int) GrB_OUTP,
                (int) GrB_MASK, (int) GrB_INP0, (int) GrB_INP1,
                (int) GxB_AxB_METHOD, (int) GxB_IMPORT) ;
    }

    return (GrB_SUCCESS) ;
}

