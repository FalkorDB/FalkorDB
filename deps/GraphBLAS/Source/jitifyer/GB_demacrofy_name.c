//------------------------------------------------------------------------------
// GB_demacrofy_name: parse a kernel name for its kname, method_code, and suffix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The kernel name has the following form, if the suffix is non-NULL:
//
//      namespace__kname__012345__suffix
//
// or, when suffix is NULL:
//
//      namespace__kname__012345
//
// where "012345" is a hexadecimal printing of the method_code.  Note the double
// underscores that precede the method_code and the suffix.
//
// GB_demacrofy_name parses the kernel_name of a PreJIT kernel, extracting the
// namespace, kname, method_code (as a uint64_t), and suffix.  NUL characters
// are inserted into kernel_name where the dots appear:
//
//      namespace._kname._012345._suffix
//
//      namespace._kname._012345.
//
// The suffix is used only for user-defined types and operators.
//
// CUDA kernels are not supported.

#include "GB.h"
#include "jitifyer/GB_stringify.h"

GrB_Info GB_demacrofy_name
(
    // input/output:
    char *kernel_name,      // string of length GB_KLEN; NUL's are inserted
                            // to demarcate each part of the kernel_name.
    // output
    char **name_space,      // namespace for the kernel_name
    char **kname,           // kname for the kernel_name
    uint64_t *method_code,  // enumify'd code of the kernel
    char **suffix           // suffix for the kernel_name (NULL if none)
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (kernel_name != NULL) ;
    ASSERT (name_space != NULL) ;
    ASSERT (kname != NULL) ;
    ASSERT (method_code != NULL) ;
    ASSERT (suffix != NULL) ;

    (*name_space) = NULL ;
    (*kname) = NULL ;
    (*method_code) = 0 ;
    (*suffix) = NULL ;  // remains NULL if kernel uses only builtin types/ops

    //--------------------------------------------------------------------------
    // find the first 2 or 3 pairs of double underscores
    //--------------------------------------------------------------------------

    size_t len = strlen (kernel_name) ;  
    if (len < 4 || len > GB_KLEN)
    { 
        // kernel_name is invalid; ignore this kernel
        return (GrB_NO_VALUE) ;     // name invalid; ignore this PreJIT kernel
    }

    int ndouble = 0 ;
    (*name_space) = kernel_name ;
    char *method_code_string = NULL ;

    for (int k = 1 ; k < len - 1 ; k++)
    {
        if (kernel_name [k-1] == '_' && kernel_name [k] == '_')
        {
            // kernel_name [k-1:k] is a double underscore: "__"
            ndouble++ ;
            // mark the end of a component of the kernel_name
            kernel_name [k-1] = '\0' ;
            // advance to the start of the next component of the kernel_name
            k++ ;
            ASSERT (kernel_name [k] != '\0') ;
            if (ndouble == 1)
            { 
                // save the start of the kname component of the kernel_name
                (*kname) = &(kernel_name [k]) ;
            }
            else if (ndouble == 2)
            { 
                // save start of the method_code component of the kernel_name
                method_code_string = &(kernel_name [k]) ;
            }
            else if (ndouble == 3)
            { 
                // save the start of the suffix component of the kernel_name
                (*suffix) = &(kernel_name [k]) ;
                // the rest of the kernel_name is the entire suffix, so quit
                break ;
            }
        }
    }

    if (ndouble < 2)
    { 
        // didn't find 2 pairs of double underscores
        // kernel_name is invalid; ignore this kernel
        return (GrB_NO_VALUE) ;     // name invald; ignore this PreJIT kernel
    }

    //--------------------------------------------------------------------------
    // parse the method_code_string
    //--------------------------------------------------------------------------

    // If this is a CUDA kernel, the method_code_string has the form 012345_72,
    // where 012345 is the method_code and 72 denotes the sm_72 CUDA
    // architecture.  The sscanf will return the method_code and the CUDA
    // architecture will be ignored.

    uint64_t method_code_result = 0 ;
    if (sscanf (method_code_string, "%" SCNx64, &method_code_result) != 1)
    { 
        // didn't find method_code: kernel_name is invalid; ignore this kernel
        return (GrB_NO_VALUE) ;     // name invald; ignore this PreJIT kernel
    }

    (*method_code) = method_code_result ;
    return (GrB_SUCCESS) ;
}

