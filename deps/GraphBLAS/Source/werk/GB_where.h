//------------------------------------------------------------------------------
// GB_where.h: definitions for Werk space and error logging
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_WHERE_H
#define GB_WHERE_H

//------------------------------------------------------------------------------
// GB_WHERE*: allocate the Werk stack and enable error logging
//------------------------------------------------------------------------------

// GB_WHERE keeps track of the currently running user-callable function.
// User-callable functions in this implementation are written so that they do
// not call other unrelated user-callable functions (except for GrB_*free).
// Related user-callable functions can call each other since they all report
// the same type-generic name.  Internal functions can be called by many
// different user-callable functions, directly or indirectly.  It would not be
// helpful to report the name of an internal function that flagged an error
// condition.  Thus, each time a user-callable function is entered, it logs the
// name of the function with the GB_WHERE macro.

#define GB_WERK(where_string)                                       \
    /* construct the Werk */                                        \
    GB_Werk_struct Werk_struct ;                                    \
    GB_Werk Werk = &Werk_struct ;                                   \
    /* set Werk->where so GrB_error can report it if needed */      \
    Werk->where = where_string ;                                    \
    /* get the pointer to where any error will be logged */         \
    Werk->logger_handle = NULL ;                                    \
    Werk->logger_size_handle = NULL ;                               \
    /* initialize the Werk stack */                                 \
    Werk->pwerk = 0 ;

// C is a matrix, vector, scalar, or descriptor
#define GB_WHERE(C,where_string)                                    \
    if (!GB_Global_GrB_init_called_get ( ))                         \
    {                                                               \
        return (GrB_PANIC) ; /* GrB_init not called */              \
    }                                                               \
    GB_WERK (where_string)                                          \
    if (C != NULL)                                                  \
    {                                                               \
        /* free any prior error logged in the object */             \
        GB_FREE (&(C->logger), C->logger_size) ;                    \
        Werk->logger_handle = &(C->logger) ;                        \
        Werk->logger_size_handle = &(C->logger_size) ;              \
    }

// create the Werk, with no error logging
#define GB_WHERE1(where_string)                                     \
    if (!GB_Global_GrB_init_called_get ( ))                         \
    {                                                               \
        return (GrB_PANIC) ; /* GrB_init not called */              \
    }                                                               \
    GB_WERK (where_string)

//------------------------------------------------------------------------------
// GB_ERROR: error logging
//------------------------------------------------------------------------------

// The GB_ERROR macro logs an error in the logger error string.
//
//  if (i >= nrows)
//  {
//      GB_ERROR (GrB_INDEX_OUT_OF_BOUNDS,
//          "Row index %d out of bounds; must be < %d", i, nrows) ;
//  }
//
// The user can then do:
//
//  const char *error ;
//  GrB_error (&error, A) ;
//  printf ("%s", error) ;

const char *GB_status_code (GrB_Info info) ;

// maximum size of the error logger string
#define GB_LOGGER_LEN 384

// log an error in the error logger string and return the error
#define GB_ERROR(info,format,...)                                           \
{                                                                           \
    if (Werk != NULL)                                                       \
    {                                                                       \
        char **logger_handle = Werk->logger_handle ;                        \
        if (logger_handle != NULL)                                          \
        {                                                                   \
            size_t *logger_size_handle = Werk->logger_size_handle ;         \
            (*logger_handle) = GB_CALLOC (GB_LOGGER_LEN+1, char,            \
                logger_size_handle) ;                                       \
            if ((*logger_handle) != NULL)                                   \
            {                                                               \
                snprintf ((*logger_handle), GB_LOGGER_LEN,                  \
                    "GraphBLAS error: %s\nfunction: %s\n" format,           \
                    GB_status_code (info), Werk->where, __VA_ARGS__) ;      \
            }                                                               \
        }                                                                   \
    }                                                                       \
    return (info) ;                                                         \
}

#endif

