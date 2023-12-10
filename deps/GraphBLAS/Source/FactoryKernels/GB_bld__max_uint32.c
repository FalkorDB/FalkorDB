//------------------------------------------------------------------------------
// GB_bld:  hard-coded functions for builder methods
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_control.h" 
#include "GB_bld__include.h"

// dup operator: Tx [k] += Sx [i], no typecast here
#define GB_BLD_DUP(Tx,k,Sx,i)  if (Sx [i] > Tx [k]) { Tx [k] = Sx [i] ; }
#define GB_BLD_COPY(Tx,k,Sx,i) Tx [k] = Sx [i]

// array types for S and T
#define GB_S_TYPE uint32_t
#define GB_T_TYPE uint32_t

// operator types: z = dup (x,y)
#define GB_Z_TYPE uint32_t
#define GB_X_TYPE uint32_t
#define GB_Y_TYPE uint32_t

// disable this operator and use the generic case if these conditions hold
#if (defined(GxB_NO_MAX) || defined(GxB_NO_UINT32) || defined(GxB_NO_MAX_UINT32))
#define GB_DISABLE 1
#else
#define GB_DISABLE 0
#endif

#include "GB_kernel_shared_definitions.h"

//------------------------------------------------------------------------------
// build a non-iso matrix
//------------------------------------------------------------------------------

GrB_Info GB (_bld__max_uint32)
(
    GB_T_TYPE *restrict Tx,
    int64_t  *restrict Ti,
    const GB_S_TYPE *restrict Sx,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    #include "GB_bld_template.c"
    return (GrB_SUCCESS) ;
    #endif
}

