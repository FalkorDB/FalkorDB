//------------------------------------------------------------------------------
// GxB_Vector_Iterator: iterate over the entries of a vector
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

#undef GxB_Vector_Iterator_getpmax
#undef GxB_Vector_Iterator_seek
#undef GxB_Vector_Iterator_next
#undef GxB_Vector_Iterator_getp
#undef GxB_Vector_Iterator_getIndex

GrB_Info GB(Vector_Iterator_bitmap_seek)
(
    GxB_Iterator iterator
)
{
    for ( ; iterator->p < iterator->pmax ; iterator->p++)
    {
        if (iterator->Ab [iterator->p])
        { 
            // found next entry
            return (GrB_SUCCESS) ;
        }
    }
    return (GxB_EXHAUSTED) ;
}

GrB_Info GxB_Vector_Iterator_attach
(
    // input/output:
    GxB_Iterator iterator,      // iterator to attach to the vector v
    // input
    GrB_Vector v,               // vector to attach
    GrB_Descriptor desc
)
{ 
    return (GB(Iterator_attach)(iterator, (GrB_Matrix) v, GxB_NO_FORMAT,
        desc)) ;
}

#if !defined ( GBMATLAB )

// These methods are not tested by the GraphBLAS/Test or GraphBLAS/Tcov
// test coverage suite, because the conflict with libmwgraphblas.so inside
// MATLAB.

uint64_t GxB_Vector_Iterator_getpmax (GxB_Iterator iterator)
{
    // return the range of the vector iterator
    return (iterator->pmax) ;
}

GrB_Info GxB_Vector_Iterator_seek (GxB_Iterator iterator, uint64_t p)
{
    // seek to a specific entry in the vector
    return (GB_Vector_Iterator_seek (iterator, p)) ;
}

GrB_Info GxB_Vector_Iterator_next (GxB_Iterator iterator)
{
    // move to the next entry of a vector
    return (GB_Vector_Iterator_next (iterator)) ;
}

uint64_t GxB_Vector_Iterator_getp (GxB_Iterator iterator)
{
    // get the current position of a vector iterator
    return (iterator->p) ;
}

uint64_t GxB_Vector_Iterator_getIndex (GxB_Iterator iterator)
{
    // get the index of a vector entry
    return (
     iterator->Ai32 ? iterator->Ai32 [iterator->p] :
    (iterator->Ai64 ? iterator->Ai64 [iterator->p] : iterator->p)) ;
}

#endif

