//------------------------------------------------------------------------------
// GB_Context.h: definitions for the GraphBLAS Context
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_CONTEXT_H
#define GB_CONTEXT_H

GrB_Info GB_Context_engage    (GxB_Context Context) ;
GrB_Info GB_Context_disengage (GxB_Context Context) ;

int    GB_Context_nthreads_max (void) ;
int    GB_Context_nthreads_max_get (GxB_Context Context) ;
void   GB_Context_nthreads_max_set (GxB_Context Context, int nthreads_max) ;

double GB_Context_chunk (void) ;
double GB_Context_chunk_get (GxB_Context Context) ;
void   GB_Context_chunk_set (GxB_Context Context, double chunk) ;

int    GB_Context_gpu_id (void) ;
int    GB_Context_gpu_id_get (GxB_Context Context) ;
void   GB_Context_gpu_id_set (GxB_Context Context, int gpu_id) ;

#endif
