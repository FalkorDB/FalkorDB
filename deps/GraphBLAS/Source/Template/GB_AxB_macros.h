//------------------------------------------------------------------------------
// GB_AxB_macros.h: macros for GB_AxB_saxpy and related methods
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_AXB_MACROS_H
#define GB_AXB_MACROS_H

// number of columns in the workspace for each task in saxpy4
#define GB_SAXPY4_PANEL_SIZE 4

// number of columns in the workspace for each saxbit task 
#define GB_SAXBIT_PANEL_SIZE 4

#define GB_SAXPY_METHOD_3 3
#define GB_SAXPY_METHOD_BITMAP 5
#define GB_SAXPY_METHOD_ISO_FULL 6

#endif

