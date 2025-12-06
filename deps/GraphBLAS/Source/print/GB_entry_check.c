//------------------------------------------------------------------------------
// GB_entry_check: print a single entry for a built-in type
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

#define GB_PRINT_INF(x) GBPR ((x < 0) ? "-Inf" : "Inf")

#define GB_PRINT_FLOAT(s)                                           \
{                                                                   \
    switch (fpclassify (s))                                         \
    {                                                               \
        case FP_NAN:      GBPR ("NaN") ; break ;                    \
        case FP_INFINITE: GB_PRINT_INF (s) ; break ;                \
        case FP_ZERO:     GBPR ("0") ; break ;                      \
        default:          GBPR ("%.6g", (double) s) ;               \
    }                                                               \
}

#define GB_PRINT_DOUBLE(d,pr_verbose)                               \
{                                                                   \
    switch (fpclassify (d))                                         \
    {                                                               \
        case FP_NAN:      GBPR ("NaN") ; break ;                    \
        case FP_INFINITE: GB_PRINT_INF (d) ; break ;                \
        case FP_ZERO:     GBPR ("0") ; break ;                      \
        default:                                                    \
            if (pr_verbose)                                         \
            {                                                       \
                /* long format */                                   \
                GBPR ("%.15g", d) ;                                 \
            }                                                       \
            else                                                    \
            {                                                       \
                /* short format */                                  \
                GBPR ("%.6g", d) ;                                  \
            }                                                       \
            break ;                                                 \
    }                                                               \
}

GrB_Info GB_entry_check     // print a single value
(
    const GrB_Type type,    // type of value to print
    const void *x,          // value to print
    int pr,                 // print level
    FILE *f,                // file to print to
    // for user-defined types only:
    char **string_handle,   // string buffer for printing
    size_t *string_size     // size of the string buffer
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL (x) ;
    GB_RETURN_IF_NULL_OR_FAULTY (type) ;

    //--------------------------------------------------------------------------
    // print the value
    //--------------------------------------------------------------------------

    const GB_Type_code code = type->code ;

    ASSERT (code <= GB_UDT_code) ;
    int64_t i ;
    uint64_t u ;
    double d ;
    float s ;
    GxB_FC32_t c ;
    GxB_FC64_t z ;
    bool pr_verbose = (pr == GxB_SHORT_VERBOSE || pr == GxB_COMPLETE_VERBOSE) ;

    switch (code)
    {

        case GB_BOOL_code   : i = *((bool     *) x) ; GBPR ("  " GBd, i) ;
            break ;
        case GB_INT8_code   : i = *((int8_t   *) x) ; GBPR ("  " GBd, i) ;
            break ;
        case GB_UINT8_code  : u = *((uint8_t  *) x) ; GBPR ("  " GBu, u) ;
            break ;
        case GB_INT16_code  : i = *((int16_t  *) x) ; GBPR ("  " GBd, i) ;
            break ;
        case GB_UINT16_code : u = *((uint16_t *) x) ; GBPR ("  " GBu, u) ;
            break ;
        case GB_INT32_code  : i = *((int32_t  *) x) ; GBPR ("  " GBd, i) ;
            break ;
        case GB_UINT32_code : u = *((uint32_t *) x) ; GBPR ("  " GBu, u) ;
            break ;
        case GB_INT64_code  : i = *((int64_t  *) x) ; GBPR ("  " GBd, i) ;
            break ;
        case GB_UINT64_code : u = *((uint64_t *) x) ; GBPR ("  " GBu, u) ;
            break ;

        case GB_FP32_code   : 
            s = *((float *) x) ;
            GBPR ("   ") ;
            GB_PRINT_FLOAT (s) ;
            break ;

        case GB_FP64_code   : 
            d = *((double *) x) ;
            GBPR ("   ") ;
            GB_PRINT_DOUBLE (d, pr_verbose) ;
            break ;

        case GB_FC32_code   : 
            c = *((GxB_FC32_t *) x) ;
            GBPR ("   ") ;
            GB_PRINT_FLOAT (GB_crealf (c)) ;
            s = GB_cimagf (c) ;
            if (s < 0)
            { 
                GBPR (" - ") ;
                GB_PRINT_FLOAT (-s) ;
            }
            else
            { 
                GBPR (" + ") ;
                GB_PRINT_FLOAT (s) ;
            }
            GBPR ("i") ;
            break ;

        case GB_FC64_code   : 
            z = *((GxB_FC64_t *) x) ;
            GBPR ("   ") ;
            GB_PRINT_DOUBLE (GB_creal (z), pr_verbose) ;
            d = GB_cimag (z) ;
            if (d < 0)
            { 
                GBPR (" - ") ;
                GB_PRINT_DOUBLE (-d, pr_verbose) ;
            }
            else
            { 
                GBPR (" + ") ;
                GB_PRINT_DOUBLE (d, pr_verbose) ;
            }
            GBPR ("i") ;
            break ;

        case GB_UDT_code    : 
            { 
                GxB_print_function pfunc = type->print_function ;
                if (pfunc != NULL
                    && string_handle != NULL && string_size != NULL)
                { 
                    // ensure the string buffer exists
                    if ((*string_handle) == NULL)
                    { 
                        // allocate the string buffer with its initial size;
                        // it is not freed here but in the caller
                        (*string_handle) = GB_MALLOC_MEMORY (1024,
                            sizeof (char), string_size) ;
                        if ((*string_handle) == NULL)
                        { 
                            return (GrB_OUT_OF_MEMORY) ;
                        }
                    }
                    for (int k = 0 ; k < 32 ; k++)
                    { 
                        int64_t result = pfunc (*string_handle, *string_size,
                            x, pr_verbose) ;
                        if (result < 0)
                        { 
                            // something went completely wrong
                            return (GrB_INVALID_VALUE) ;
                        }
                        else if (result >= (*string_size))
                        { 
                            // string is too small; make it bigger
                            size_t newsize = GB_IMAX (result+2,
                                2 * (*string_size)) ;
                            bool ok = true ;
                            GB_REALLOC_MEMORY ((*string_handle), newsize,
                                sizeof (char), string_size, &ok) ;
                            if (!ok)
                            { 
                                // out of memory
                                return (GrB_OUT_OF_MEMORY) ;
                            }
                            (*string_size) = newsize ;
                        }
                        else
                        { 
                            // success
                            break ;
                        }
                    }
                    // print the string (also ensure it is NUL terminated)
                    char *string = (*string_handle) ;
                    string [(*string_size)-1] = '\0' ;
                    GBPR ("   %s", string) ;
                }
                else
                { 
                    // no print function registered for this type, or no
                    // place to put the string buffer
                    GBPR ("[user-defined value]") ;
                }
            }
            break ;
        default: ;
    }

    return (GrB_SUCCESS) ;
}

