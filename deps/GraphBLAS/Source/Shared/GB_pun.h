//------------------------------------------------------------------------------
// GB_pun.h: type punning
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// With type punning, a lvalue is treated as a different type, but with no
// typecasting.  The address of the lvalue is first typecasted to a (type *)
// pointer, and then the pointer is dereferenced.  The lvalue must not be an
// expression; it must be possible to take its address, as &lvalue.

#ifndef GB_PUN_H
#define GB_PUN_H

#undef  GB_PUN
#define GB_PUN(type,value) (*((type *) (&(value))))

#endif

