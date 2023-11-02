//------------------------------------------------------------------------------
// GB_ok.h: macros for checking inputs and returning if an error occurs
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_OK_H
#define GB_OK_H

//------------------------------------------------------------------------------
// GB_OK: call a method and take corrective active if it fails
//------------------------------------------------------------------------------

#define GB_OK(method)                       \
{                                           \
    info = method ;                         \
    if (info != GrB_SUCCESS)                \
    {                                       \
        GB_FREE_ALL ;                       \
        return (info) ;                     \
    }                                       \
}

//------------------------------------------------------------------------------
// GB_RETURN_*: input guards for user-callable GrB* and GxB* methods
//------------------------------------------------------------------------------

// check if a required arg is NULL
#define GB_RETURN_IF_NULL(arg)                                          \
    if ((arg) == NULL)                                                  \
    {                                                                   \
        /* the required arg is NULL */                                  \
        return (GrB_NULL_POINTER) ;                                     \
    }

// arg may be NULL, but if non-NULL then it must be initialized
#define GB_RETURN_IF_FAULTY(arg)                                        \
    if ((arg) != NULL && (arg)->magic != GB_MAGIC)                      \
    {                                                                   \
        if ((arg)->magic == GB_MAGIC2)                                  \
        {                                                               \
            /* optional arg is not NULL, but invalid */                 \
            return (GrB_INVALID_OBJECT) ;                               \
        }                                                               \
        else                                                            \
        {                                                               \
            /* optional arg is not NULL, but not initialized */         \
            return (GrB_UNINITIALIZED_OBJECT) ;                         \
        }                                                               \
    }

// arg must not be NULL, and it must be initialized
#define GB_RETURN_IF_NULL_OR_FAULTY(arg)                                \
    GB_RETURN_IF_NULL (arg) ;                                           \
    GB_RETURN_IF_FAULTY (arg) ;

// positional ops not supported for use as accum operators
#define GB_RETURN_IF_FAULTY_OR_POSITIONAL(accum)                        \
{                                                                       \
    GB_RETURN_IF_FAULTY (accum) ;                                       \
    if (GB_OP_IS_POSITIONAL (accum))                                    \
    {                                                                   \
        GB_ERROR (GrB_DOMAIN_MISMATCH,                                  \
            "Positional op z=%s(x,y) not supported as accum\n",         \
                accum->name) ;                                          \
    }                                                                   \
}

// C<M>=Z ignores Z if an empty mask is complemented, or if M is full,
// structural and complemented, so return from the method without computing
// anything.  Clear C if replace option is true.
#define GB_RETURN_IF_QUICK_MASK(C, C_replace, M, Mask_comp, Mask_struct)    \
    if (Mask_comp && (M == NULL || (GB_IS_FULL (M) && Mask_struct)))        \
    {                                                                       \
        /* C<!NULL>=NULL since result does not depend on computing Z */     \
        return (C_replace ? GB_clear (C, Werk) : GrB_SUCCESS) ;          \
    }

//------------------------------------------------------------------------------
// GB_GET_DESCRIPTOR*: get the contents of a descriptor
//------------------------------------------------------------------------------

// check the descriptor and extract its contents
#define GB_GET_DESCRIPTOR(info,desc,dout,dmc,dms,d0,d1,dalgo,dsort)          \
    GrB_Info info ;                                                          \
    bool dout, dmc, dms, d0, d1 ;                                            \
    int dsort ;                                                              \
    GrB_Desc_Value dalgo ;                                                   \
    /* if desc is NULL then defaults are used.  This is OK */                \
    info = GB_Descriptor_get (desc, &dout, &dmc, &dms, &d0, &d1, &dalgo,     \
        &dsort) ;                                                            \
    if (info != GrB_SUCCESS)                                                 \
    {                                                                        \
        /* desc not NULL, but uninitialized or an invalid object */          \
        return (info) ;                                                      \
    }

#define GB_GET_DESCRIPTOR_IMPORT(desc,fast_import)                          \
    /* default is a fast import, where the data is trusted */               \
    bool fast_import = true ;                                               \
    if (desc != NULL && desc->import != GxB_FAST_IMPORT)                    \
    {                                                                       \
        /* input data is not trusted */                                     \
        fast_import = false ;                                               \
    }

//------------------------------------------------------------------------------
// GB_VECTOR_OK, GB_SCALAR_OK: check if typecast from GrB_Matrix is OK
//------------------------------------------------------------------------------

// The internal content of a GrB_Matrix and GrB_Vector are identical, and
// inside SuiteSparse:GraphBLAS, they can be typecasted between each other.
// This typecasting feature should not be done in user code, however, since it
// is not supported in the API.  All GrB_Vector objects can be safely
// typecasted into a GrB_Matrix, but not the other way around.  The GrB_Vector
// object is more restrictive.  The GB_VECTOR_OK(v) macro defines the content
// that all GrB_Vector objects must have.

// GB_VECTOR_OK(v) is used mainly for assertions, but also to determine when it
// is safe to typecast an n-by-1 GrB_Matrix (in standard CSC format) into a
// GrB_Vector.  This is not done in the main SuiteSparse:GraphBLAS library, but
// in the GraphBLAS/Test directory only.  The macro is also used in
// GB_Vector_check, to ensure the content of a GrB_Vector is valid.

#define GB_VECTOR_OK(v)                     \
(                                           \
    ((v) != NULL) &&                        \
    ((v)->is_csc == true) &&                \
    ((v)->plen == 1 || (v)->plen == -1) &&  \
    ((v)->vdim == 1) &&                     \
    ((v)->nvec == 1) &&                     \
    ((v)->h == NULL)                        \
)

// A GxB_Vector is a GrB_Vector of length 1
#define GB_SCALAR_OK(v) (GB_VECTOR_OK(v) && ((v)->vlen == 1))

#endif

