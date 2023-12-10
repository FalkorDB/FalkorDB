//------------------------------------------------------------------------------
// GrB_Descriptor_set_*: set a field in a descriptor
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_get_set.h"

//------------------------------------------------------------------------------
// GB_desc_set
//------------------------------------------------------------------------------

static GrB_Info GB_desc_set
(
    GrB_Descriptor desc,        // descriptor to modify
    int32_t value,              // value to change it to
    int field,                  // parameter to change
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // set the parameter
    //--------------------------------------------------------------------------

    int mask = (int) desc->mask ;

    switch (field)
    {

        case GrB_OUTP :             // same as GrB_OUTP_FIELD

            if (! (value == GrB_DEFAULT || value == GrB_REPLACE))
            { 
                GB_ERROR (GrB_INVALID_VALUE,
                        "invalid descriptor value [%d] for GrB_OUTP field;\n"
                        "must be GrB_DEFAULT [%d] or GrB_REPLACE [%d]",
                        value, (int) GrB_DEFAULT, (int) GrB_REPLACE) ;
            }
            desc->out = (GrB_Desc_Value) value ;
            break ;

        case GrB_MASK :             // same as GrB_MASK_FIELD

            if (! (value == GrB_DEFAULT ||
                   value == GrB_COMP ||
                   value == GrB_STRUCTURE ||
                   value == (GrB_COMP + GrB_STRUCTURE)))
            { 
                GB_ERROR (GrB_INVALID_VALUE,
                        "invalid descriptor value [%d] for GrB_MASK field;\n"
                        "must be GrB_DEFAULT [%d], GrB_COMP [%d],\n"
                        "GrB_STRUCTURE [%d], or GrB_COMP+GrB_STRUCTURE [%d]",
                        value, (int) GrB_DEFAULT, (int) GrB_COMP,
                        (int) GrB_STRUCTURE,
                        (int) (GrB_COMP + GrB_STRUCTURE)) ;
            }
            switch (value)
            {
                case GrB_COMP      : mask |= GrB_COMP ;      break ;
                case GrB_STRUCTURE : mask |= GrB_STRUCTURE ; break ;
                default            : mask = value ;          break ;
            }
            desc->mask = (GrB_Desc_Value) mask ;
            break ;

        case GrB_INP0 :             // same as GrB_INP0_FIELD

            if (! (value == GrB_DEFAULT || value == GrB_TRAN))
            { 
                GB_ERROR (GrB_INVALID_VALUE,
                        "invalid descriptor value [%d] for GrB_INP0 field;\n"
                        "must be GrB_DEFAULT [%d] or GrB_TRAN [%d]",
                        value, (int) GrB_DEFAULT, (int) GrB_TRAN) ;
            }
            desc->in0 = (GrB_Desc_Value) value ;
            break ;

        case GrB_INP1 :             // same as GrB_INP1_FIELD

            if (! (value == GrB_DEFAULT || value == GrB_TRAN))
            { 
                GB_ERROR (GrB_INVALID_VALUE,
                        "invalid descriptor value [%d] for GrB_INP1 field;\n"
                        "must be GrB_DEFAULT [%d] or GrB_TRAN [%d]",
                        value, (int) GrB_DEFAULT, (int) GrB_TRAN) ;
            }
            desc->in1 = (GrB_Desc_Value) value ;
            break ;

        case GxB_AxB_METHOD : 

            if (! (value == GrB_DEFAULT  || value == GxB_AxB_GUSTAVSON
                || value == GxB_AxB_DOT
                || value == GxB_AxB_HASH || value == GxB_AxB_SAXPY))
            { 
                GB_ERROR (GrB_INVALID_VALUE,
                        "invalid descriptor value [%d] for GrB_AxB_METHOD"
                        " field;\nmust be GrB_DEFAULT [%d], GxB_AxB_GUSTAVSON"
                        " [%d]\nGxB_AxB_DOT [%d]"
                        " GxB_AxB_HASH [%d] or GxB_AxB_SAXPY [%d]",
                        value, (int) GrB_DEFAULT, (int) GxB_AxB_GUSTAVSON,
                        (int) GxB_AxB_DOT,
                        (int) GxB_AxB_HASH, (int) GxB_AxB_SAXPY) ;
            }
            desc->axb = (GrB_Desc_Value) value ;
            break ;

        case GxB_SORT : 

            desc->do_sort = value ;
            break ;

        case GxB_COMPRESSION : 

            desc->compression = value ;
            break ;

        case GxB_IMPORT : 

            // In case the user application does not check the return value
            // of this method, an error condition is never returned.
            desc->import =
                (value == GrB_DEFAULT) ? GxB_FAST_IMPORT : GxB_SECURE_IMPORT ;
            break ;

        default : 
            return (GrB_INVALID_VALUE) ;
    }

    #pragma omp flush
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GrB_Descriptor_set_Scalar
//------------------------------------------------------------------------------

GrB_Info GrB_Descriptor_set_Scalar
(
    GrB_Descriptor desc,
    GrB_Scalar value,
    GrB_Field field
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (desc == NULL || desc->header_size == 0)
    { 
        // built-in descriptors may not be modified
        return (GrB_INVALID_VALUE) ;
    }

    GB_WHERE (desc, "GrB_Descriptor_set_Scalar (desc, value, field)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (desc) ;
    GB_RETURN_IF_NULL_OR_FAULTY (value) ;
    ASSERT_DESCRIPTOR_OK (desc, "desc to set", GB0) ;

    //--------------------------------------------------------------------------
    // set the field
    //--------------------------------------------------------------------------

    int32_t i ;
    GrB_Info info = GrB_Scalar_extractElement_INT32 (&i, value) ;
    if (info != GrB_SUCCESS)
    { 
        return ((info == GrB_NO_VALUE) ? GrB_EMPTY_OBJECT : info) ;
    } 
    return (GB_desc_set (desc, i, field, Werk)) ;
}

//------------------------------------------------------------------------------
// GrB_Descriptor_set_String
//------------------------------------------------------------------------------

GrB_Info GrB_Descriptor_set_String
(
    GrB_Descriptor desc,
    char * value,
    GrB_Field field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (desc == NULL || desc->header_size == 0 || field != GrB_NAME)
    { 
        // built-in descriptors may not be modified
        return (GrB_INVALID_VALUE) ;
    }

    GB_WHERE (desc, "GrB_Descriptor_set_String (desc, value, field)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (desc) ;
    GB_RETURN_IF_NULL (value) ;
    ASSERT_DESCRIPTOR_OK (desc, "desc to set", GB0) ;

    //--------------------------------------------------------------------------
    // set the field
    //--------------------------------------------------------------------------

    return (GB_user_name_set (&(desc->user_name), &(desc->user_name_size),
        value, false)) ;
}

//------------------------------------------------------------------------------
// GrB_Descriptor_set_INT32
//------------------------------------------------------------------------------

GrB_Info GrB_Descriptor_set_INT32
(
    GrB_Descriptor desc,
    int32_t value,
    GrB_Field field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (desc == NULL || desc->header_size == 0)
    { 
        // built-in descriptors may not be modified
        return (GrB_INVALID_VALUE) ;
    }

    GB_WHERE (desc, "GrB_Descriptor_set_INT32 (desc, value, field)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (desc) ;
    ASSERT_DESCRIPTOR_OK (desc, "desc to set", GB0) ;

    //--------------------------------------------------------------------------
    // set the field
    //--------------------------------------------------------------------------

    return (GB_desc_set (desc, value, field, Werk)) ;
}

//------------------------------------------------------------------------------
// GrB_Descriptor_set_VOID
//------------------------------------------------------------------------------

GrB_Info GrB_Descriptor_set_VOID
(
    GrB_Descriptor desc,
    void * value,
    GrB_Field field,
    size_t size
)
{ 
    return (GrB_INVALID_VALUE) ;
}

