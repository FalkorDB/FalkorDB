//------------------------------------------------------------------------------
// GB_werk.h: definitions for werkspace management on the Werk stack
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_WERK_H
#define GB_WERK_H

//------------------------------------------------------------------------------
// GB_Werk: error logging and Werk space
//------------------------------------------------------------------------------

// Error messages are logged in Werk->logger_handle, on the stack which is
// handle to the input/output matrix/vector (typically C).  If the user-defined
// data types, operators, etc have really long names, the error messages are
// safely truncated (via snprintf).  This is intentional, but gcc with
// -Wformat-truncation will print a warning (see pragmas above).  Ignore the
// warning.

// GB_WERK_SIZE is the size of a small fixed-sized array in the Werk, used
// for small werkspace allocations (typically O(# of threads or # tasks)).
// GB_WERK_SIZE must be a multiple of 8.  The Werk->Stack array is placed first
// in the GB_Werk struct, to ensure proper alignment.

#define GB_WERK_SIZE 16384

typedef struct
{
    GB_void Stack [GB_WERK_SIZE] ;  // werkspace stack
    const char *where ;             // GraphBLAS function where error occurred
    char **logger_handle ;          // error report
    size_t *logger_size_handle ;
    int pwerk ;                     // top of Werk stack, initially zero
}
GB_Werk_struct ;

typedef GB_Werk_struct *GB_Werk ;

//------------------------------------------------------------------------------
// GB_werk_push/pop: manage werkspace in the Werk stack
//------------------------------------------------------------------------------

// Werk->Stack is a small fixed-size array that is allocated on the stack
// of any user-callable GraphBLAS function.  It is used for small werkspace
// allocations.

// GB_ROUND8(s) rounds up s to a multiple of 8
#define GB_ROUND8(s) (((s) + 7) & (~0x7))

//------------------------------------------------------------------------------
// GB_werk_push: allocate werkspace from the Werk stack or malloc
//------------------------------------------------------------------------------

// The werkspace is allocated from the Werk static if it small enough and space
// is available.  Otherwise it is allocated by malloc.

// See GB_callbacks.h for the prototype.

//------------------------------------------------------------------------------
// Werk helper macros
//------------------------------------------------------------------------------

// declare a werkspace X of a given type
#define GB_WERK_DECLARE(X,type)                             \
    type *restrict X = NULL ;                               \
    bool X ## _on_stack = false ;                           \
    size_t X ## _nitems = 0, X ## _size_allocated = 0 ;

// push werkspace X
#define GB_WERK_PUSH(X,nitems,type)                                         \
    X ## _nitems = (nitems) ;                                               \
    X = (type *) GB_werk_push (&(X ## _size_allocated), &(X ## _on_stack),  \
        X ## _nitems, sizeof (type), Werk) ; 

// pop werkspace X
#define GB_WERK_POP(X,type)                                                 \
    X = (type *) GB_werk_pop (X, &(X ## _size_allocated), X ## _on_stack,   \
        X ## _nitems, sizeof (type), Werk) ; 

//------------------------------------------------------------------------------
// GB_werk_pop:  free werkspace from the Werk stack
//------------------------------------------------------------------------------

// If the werkspace was allocated from the Werk stack, it must be at the top of
// the stack to free it properly.  Freeing a werkspace in the middle of the
// Werk stack also frees everything above it.  This is not a problem if that
// space is also being freed, but the assertion below ensures that the freeing
// werkspace from the Werk stack is done in LIFO order, like a stack.

// See GB_callbacks.h for the prototype.

#endif

