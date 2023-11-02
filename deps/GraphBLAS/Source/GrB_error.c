//------------------------------------------------------------------------------
// GrB_error: return an error string describing the last error
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

const char empty [8] = "" ;

GrB_Info GxB_Context_error (const char **error, const GxB_Context context)
{ 
    GB_RETURN_IF_NULL (error) ;
    (*error) = empty ;
    return (GrB_SUCCESS) ;
}

GrB_Info GrB_Type_error (const char **error, const GrB_Type type)
{ 
    GB_RETURN_IF_NULL (error) ;
    (*error) = empty ;
    return (GrB_SUCCESS) ;
}

GrB_Info GrB_UnaryOp_error (const char **error, const GrB_UnaryOp op)
{ 
    GB_RETURN_IF_NULL (error) ;
    (*error) = empty ;
    return (GrB_SUCCESS) ;
}

GrB_Info GrB_BinaryOp_error (const char **error, const GrB_BinaryOp op)
{ 
    GB_RETURN_IF_NULL (error) ;
    (*error) = empty ;
    return (GrB_SUCCESS) ;
}

GrB_Info GrB_IndexUnaryOp_error (const char **error, const GrB_IndexUnaryOp op)
{ 
    GB_RETURN_IF_NULL (error) ;
    (*error) = empty ;
    return (GrB_SUCCESS) ;
}

GrB_Info GrB_Monoid_error (const char **error, const GrB_Monoid monoid)
{ 
    GB_RETURN_IF_NULL (error) ;
    (*error) = empty ;
    return (GrB_SUCCESS) ;
}

GrB_Info GrB_Semiring_error (const char **error, const GrB_Semiring semiring)
{ 
    GB_RETURN_IF_NULL (error) ;
    (*error) = empty ;
    return (GrB_SUCCESS) ;
}

GrB_Info GrB_Scalar_error (const char **error, const GrB_Scalar s)
{ 
    GB_RETURN_IF_NULL (error) ;
    (*error) = empty ;
    GB_RETURN_IF_NULL_OR_FAULTY (s) ;
    if (s->logger == NULL) return (GrB_SUCCESS) ;
    (*error) = s->logger ;
    return (GrB_SUCCESS) ;
}

// historical; use GrB_Scalar_error instead.
GrB_Info GxB_Scalar_error (const char **error, const GrB_Scalar s)
{
    return (GrB_Scalar_error (error, s)) ;
}

GrB_Info GrB_Vector_error (const char **error, const GrB_Vector v)
{ 
    GB_RETURN_IF_NULL (error) ;
    (*error) = empty ;
    GB_RETURN_IF_NULL_OR_FAULTY (v) ;
    if (v->logger == NULL) return (GrB_SUCCESS) ;
    (*error) = v->logger ;
    return (GrB_SUCCESS) ;
}

GrB_Info GrB_Matrix_error (const char **error, const GrB_Matrix A)
{ 
    GB_RETURN_IF_NULL (error) ;
    (*error) = empty ;
    GB_RETURN_IF_NULL_OR_FAULTY (A) ;
    if (A->logger == NULL) return (GrB_SUCCESS) ;
    (*error) = A->logger ;
    return (GrB_SUCCESS) ;
}

GrB_Info GrB_Descriptor_error (const char **error, const GrB_Descriptor d)
{ 
    GB_RETURN_IF_NULL (error) ;
    (*error) = empty ;
    GB_RETURN_IF_NULL_OR_FAULTY (d) ;
    if (d->logger == NULL) return (GrB_SUCCESS) ;
    (*error) = d->logger ;
    return (GrB_SUCCESS) ;
}

