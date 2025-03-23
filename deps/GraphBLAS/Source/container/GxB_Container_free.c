//------------------------------------------------------------------------------
// GxB_Container_free: free a Container
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_container.h"

GrB_Info GxB_Container_free
(
    GxB_Container *Container
)
{ 

    if (Container != NULL && (*Container) != NULL)
    { 

        //----------------------------------------------------------------------
        // free each GrB_Vector component of the Container
        //----------------------------------------------------------------------

        GB_Matrix_free ((GrB_Matrix *) &((*Container)->p)) ;
        GB_Matrix_free ((GrB_Matrix *) &((*Container)->h)) ;
        GB_Matrix_free ((GrB_Matrix *) &((*Container)->b)) ;
        GB_Matrix_free ((GrB_Matrix *) &((*Container)->i)) ;
        GB_Matrix_free ((GrB_Matrix *) &((*Container)->x)) ;

        //----------------------------------------------------------------------
        // free each GrB_Matrix component of the Container
        //----------------------------------------------------------------------

        GB_Matrix_free (&((*Container)->Y)) ;

        //----------------------------------------------------------------------
        // free the Container itself
        //----------------------------------------------------------------------

        GB_FREE_MEMORY (Container, sizeof (struct GxB_Container_struct)) ;
    }

    return (GrB_SUCCESS) ;
}

