//------------------------------------------------------------------------------
// GB_macrofy_cast_expression: construct a typecasting string
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Return a typecast expression to cast from xtype to ztype.

#include "GB.h"
#include "GB_stringify.h"

const char *GB_macrofy_cast_expression  // return cast expression
(
    FILE *fp,
    // input:
    GrB_Type ztype,     // output type
    GrB_Type xtype,     // input type
    // output
    int *nargs          // # of string arguments in output format
)
{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    (*nargs) = 2 ;
    const char *f = NULL ;
    ASSERT (ztype != NULL) ;
    ASSERT (xtype != NULL) ;
    const GB_Type_code zcode = ztype->code ;
    const GB_Type_code xcode = xtype->code ;

    if (zcode == xcode)
    { 

        //----------------------------------------------------------------------
        // no typecasting
        //----------------------------------------------------------------------

        // user-defined types come here, and require type->defn and type->name
        // to both be defined.  If the user-defined type has no name or defn,
        // then no JIT kernel can be created for it.

        ASSERT (GB_IMPLIES (zcode == GB_UDT_code, (ztype == xtype) &&
            ztype->name != NULL && ztype->defn != NULL)) ;

        f = "%s = %s" ;

    }
    else if (zcode == GB_BOOL_code)
    {

        //----------------------------------------------------------------------
        // typecast to boolean
        //----------------------------------------------------------------------

        if (xcode == GB_FC32_code)
        { 
            f = "%s = (GB_crealf (%s) != 0 || GB_cimagf (%s) != 0)" ;
            (*nargs) = 3 ;
        }
        else if (xcode == GB_FC64_code)
        { 
            f = "%s = (GB_creal (%s) != 0 || GB_cimag (%s) != 0)" ;
            (*nargs) = 3 ;
        }
        else
        { 
            f = "%s = ((%s) != 0)" ;
        }

    }
    else if ((zcode >= GB_INT8_code && zcode <= GB_UINT64_code)
          && (xcode >= GB_FP32_code && zcode <= GB_FC64_code))
    {

        //----------------------------------------------------------------------
        // typecast to integer from floating-point
        //----------------------------------------------------------------------

        switch (zcode)
        {
            case GB_INT8_code   : 

                switch (xcode)
                {
                    case GB_FP32_code : 
                        f = "%s = GJ_cast_to_int8 ((double) (%s))" ;
                        break ;
                    case GB_FP64_code : 
                        f = "%s = GJ_cast_to_int8 (%s)" ;
                        break ;
                    case GB_FC32_code : 
                        f = "%s = GJ_cast_to_int8 ((double) GB_crealf (%s))" ;
                        break ;
                    case GB_FC64_code : 
                        f = "%s = GJ_cast_to_int8 (GB_creal (%s))" ;
                        break ;
                    default:;
                }
                GB_macrofy_defn (fp, 0, "GJ_cast_to_int8",
                    GJ_cast_to_int8_DEFN) ;
                break ;

            case GB_INT16_code  : 

                switch (xcode)
                {
                    case GB_FP32_code : 
                        f = "%s = GJ_cast_to_int16 ((double) (%s))" ;
                        break ;
                    case GB_FP64_code : 
                        f = "%s = GJ_cast_to_int16 (%s)" ;
                        break ;
                    case GB_FC32_code : 
                        f = "%s = GJ_cast_to_int16 ((double) GB_crealf (%s))" ;
                        break ;
                    case GB_FC64_code : 
                        f = "%s = GJ_cast_to_int16 (GB_creal (%s))" ;
                        break ;
                    default:;
                }
                GB_macrofy_defn (fp, 0, "GJ_cast_to_int16",
                    GJ_cast_to_int16_DEFN) ;
                break ;

            case GB_INT32_code  : 

                switch (xcode)
                {
                    case GB_FP32_code : 
                        f = "%s = GJ_cast_to_int32 ((double) (%s))" ;
                        break ;
                    case GB_FP64_code : 
                        f = "%s = GJ_cast_to_int32 (%s)" ;
                        break ;
                    case GB_FC32_code : 
                        f = "%s = GJ_cast_to_int32 ((double) GB_crealf (%s))" ;
                        break ;
                    case GB_FC64_code : 
                        f = "%s = GJ_cast_to_int32 (GB_creal (%s))" ;
                        break ;
                    default:;
                }
                GB_macrofy_defn (fp, 0, "GJ_cast_to_int32",
                    GJ_cast_to_int32_DEFN) ;
                break ;

            case GB_INT64_code  : 

                switch (xcode)
                {
                    case GB_FP32_code : 
                        f = "%s = GJ_cast_to_int64 ((double) (%s))" ;
                        break ;
                    case GB_FP64_code : 
                        f = "%s = GJ_cast_to_int64 (%s)" ;
                        break ;
                    case GB_FC32_code : 
                        f = "%s = GJ_cast_to_int64 ((double) GB_crealf (%s))" ;
                        break ;
                    case GB_FC64_code : 
                        f = "%s = GJ_cast_to_int64 (GB_creal (%s))" ;
                        break ;
                    default:;
                }
                GB_macrofy_defn (fp, 0, "GJ_cast_to_int64",
                    GJ_cast_to_int64_DEFN) ;
                break ;

            case GB_UINT8_code  : 

                switch (xcode)
                {
                    case GB_FP32_code : 
                        f = "%s = GJ_cast_to_uint8 ((double) (%s))" ;
                        break ;
                    case GB_FP64_code : 
                        f = "%s = GJ_cast_to_uint8 (%s)" ;
                        break ;
                    case GB_FC32_code : 
                        f = "%s = GJ_cast_to_uint8 ((double) GB_crealf (%s))" ;
                        break ;
                    case GB_FC64_code : 
                        f = "%s = GJ_cast_to_uint8 (GB_creal (%s))" ;
                        break ;
                    default:;
                }
                GB_macrofy_defn (fp, 0, "GJ_cast_to_uint8",
                    GJ_cast_to_uint8_DEFN) ;
                break ;

            case GB_UINT16_code : 

                switch (xcode)
                {
                    case GB_FP32_code : 
                        f = "%s = GJ_cast_to_uint16 ((double) (%s))" ;
                        break ;
                    case GB_FP64_code : 
                        f = "%s = GJ_cast_to_uint16 (%s)" ;
                        break ;
                    case GB_FC32_code : 
                        f = "%s = GJ_cast_to_uint16 ((double) GB_crealf (%s))" ;
                        break ;
                    case GB_FC64_code : 
                        f = "%s = GJ_cast_to_uint16 (GB_creal (%s))" ;
                        break ;
                    default:;
                }
                GB_macrofy_defn (fp, 0, "GJ_cast_to_uint16",
                    GJ_cast_to_uint16_DEFN) ;
                break ;

            case GB_UINT32_code : 

                switch (xcode)
                {
                    case GB_FP32_code : 
                        f = "%s = GJ_cast_to_uint32 ((double) (%s))" ;
                        break ;
                    case GB_FP64_code : 
                        f = "%s = GJ_cast_to_uint32 (%s)" ;
                        break ;
                    case GB_FC32_code : 
                        f = "%s = GJ_cast_to_uint32 ((double) GB_crealf (%s))" ;
                        break ;
                    case GB_FC64_code : 
                        f = "%s = GJ_cast_to_uint32 (GB_creal (%s))" ;
                        break ;
                    default:;
                }
                GB_macrofy_defn (fp, 0, "GJ_cast_to_uint32",
                    GJ_cast_to_uint32_DEFN) ;
                break ;

            case GB_UINT64_code : 

                switch (xcode)
                {
                    case GB_FP32_code : 
                        f = "%s = GJ_cast_to_uint64 ((double) (%s))" ;
                        break ;
                    case GB_FP64_code : 
                        f = "%s = GJ_cast_to_uint64 (%s)" ;
                        break ;
                    case GB_FC32_code : 
                        f = "%s = GJ_cast_to_uint64 ((double) GB_crealf (%s))" ;
                        break ;
                    case GB_FC64_code : 
                        f = "%s = GJ_cast_to_uint64 (GB_creal (%s))" ;
                        break ;
                    default:;
                }
                GB_macrofy_defn (fp, 0, "GJ_cast_to_uint64",
                    GJ_cast_to_uint64_DEFN) ;
                break ;

            default:;
        }

    }
    else
    { 

        //----------------------------------------------------------------------
        // all other cases: use C11 typecasting rules
        //----------------------------------------------------------------------

        f = NULL ;
        (*nargs) = 0 ;

    }

    return (f) ;
}

