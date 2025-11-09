//------------------------------------------------------------------------------
// GB_where.h: definitions for Werk space and error logging
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_WHERE_H
#define GB_WHERE_H

//------------------------------------------------------------------------------
// GB_Werk_init: initialize the Werk object
//------------------------------------------------------------------------------

static inline GB_Werk GB_Werk_init (GB_Werk Werk, const char *where_string)
{
    // set Werk->where so GrB_error can report it if needed
    Werk->where = where_string ;

    // get the pointer to where any error will be logged
    Werk->logger_handle = NULL ;
    Werk->logger_size_handle = NULL ;

    // initialize the Werk stack
    Werk->pwerk = 0 ;

    // get the global integer control; revised with C->[pji]_control by
    // GB_WHERE_C_LOGGER (C).
    Werk->p_control = GB_Global_p_control_get ( ) ;
    Werk->j_control = GB_Global_j_control_get ( ) ;
    Werk->i_control = GB_Global_i_control_get ( ) ;

    // return result
    return (Werk) ;
}

//------------------------------------------------------------------------------
// GB_valids: return GrB_SUCCESS if matrices are valid, error otherwise
//------------------------------------------------------------------------------

#define GB_RETURN_IF_INVALID(arg)               \
    info = GB_valid_matrix ((GrB_Matrix) arg) ; \
    if (info != GrB_SUCCESS)                    \
    {                                           \
        return (info) ;                         \
    }

static inline GrB_Info GB_valid6
(
    void *arg1,
    void *arg2,
    void *arg3,
    void *arg4,
    void *arg5,
    void *arg6
)
{
    GrB_Info info ;
    GB_RETURN_IF_INVALID (arg1) ;
    GB_RETURN_IF_INVALID (arg2) ;
    GB_RETURN_IF_INVALID (arg3) ;
    GB_RETURN_IF_INVALID (arg4) ;
    GB_RETURN_IF_INVALID (arg5) ;
    GB_RETURN_IF_INVALID (arg6) ;
    return (GrB_SUCCESS) ;
}

static inline GrB_Info GB_valid5
(
    void *arg1,
    void *arg2,
    void *arg3,
    void *arg4,
    void *arg5
)
{
    GrB_Info info ;
    GB_RETURN_IF_INVALID (arg1) ;
    GB_RETURN_IF_INVALID (arg2) ;
    GB_RETURN_IF_INVALID (arg3) ;
    GB_RETURN_IF_INVALID (arg4) ;
    GB_RETURN_IF_INVALID (arg5) ;
    return (GrB_SUCCESS) ;
}

static inline GrB_Info GB_valid4
(
    void *arg1,
    void *arg2,
    void *arg3,
    void *arg4
)
{
    GrB_Info info ;
    GB_RETURN_IF_INVALID (arg1) ;
    GB_RETURN_IF_INVALID (arg2) ;
    GB_RETURN_IF_INVALID (arg3) ;
    GB_RETURN_IF_INVALID (arg4) ;
    return (GrB_SUCCESS) ;
}

static inline GrB_Info GB_valid3
(
    void *arg1,
    void *arg2,
    void *arg3
)
{
    GrB_Info info ;
    GB_RETURN_IF_INVALID (arg1) ;
    GB_RETURN_IF_INVALID (arg2) ;
    GB_RETURN_IF_INVALID (arg3) ;
    return (GrB_SUCCESS) ;
}

static inline GrB_Info GB_valid2
(
    void *arg1,
    void *arg2
)
{
    GrB_Info info ;
    GB_RETURN_IF_INVALID (arg1) ;
    GB_RETURN_IF_INVALID (arg2) ;
    return (GrB_SUCCESS) ;
}

static inline GrB_Info GB_valid1
(
    void *arg1
)
{
    GrB_Info info ;
    GB_RETURN_IF_INVALID (arg1) ;
    return (GrB_SUCCESS) ;
}

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

#define GB_CHECK_INIT                                               \
    if (!GB_Global_GrB_init_called_get ( ))                         \
    {                                                               \
        return (GrB_PANIC) ; /* GrB_init not called */              \
    }                                                               \

#define GB_WERK(where_string)                                       \
    /* construct the Werk */                                        \
    GB_Werk_struct Werk_struct ;                                    \
    GB_Werk Werk = GB_Werk_init (&Werk_struct, where_string) ;

// create the Werk, with no error logging
#define GB_WHERE0(where_string)                                     \
    GrB_Info info ;                                                 \
    GB_CHECK_INIT                                                   \
    GB_WERK (where_string)

// C is a matrix, vector, or scalar
#define GB_WHERE(C,arg2,arg3,arg4,arg5,arg6,where_string)           \
    GB_WHERE0 (where_string)                                        \
    /* ensure the matrix has valid integers */                      \
    info = GB_valid6 (C, arg2, arg3, arg4, arg5, arg6) ;            \
    GB_WHERE_C_LOGGER (C)

#define GB_WHERE_CHECK_INFO                                         \
    if (info != GrB_SUCCESS)                                        \
    {                                                               \
        return (info) ;                                             \
    }                                                               \

#define GB_WHERE_C_LOGGER(C)                                        \
    GB_WHERE_CHECK_INFO                                             \
    if (C != NULL)                                                  \
    {                                                               \
        /* free any prior error logged in the object */             \
        GB_FREE_MEMORY (&(C->logger), C->logger_size) ;             \
        /* get the error logger */                                  \
        Werk->logger_handle = &(C->logger) ;                        \
        Werk->logger_size_handle = &(C->logger_size) ;              \
        /* combine the matrix and global pji_control */             \
        Werk->p_control = GB_pji_control (C->p_control, Werk->p_control) ; \
        Werk->j_control = GB_pji_control (C->j_control, Werk->j_control) ; \
        Werk->i_control = GB_pji_control (C->i_control, Werk->i_control) ; \
    }

// GB_WHEREn: check n arguments, first one is input/output matrix C for logger
#define GB_WHERE6(C,arg2,arg3,arg4,arg5,arg6,where_string)          \
    GB_WHERE0 (where_string)                                        \
    info = GB_valid6 (C, arg2, arg3, arg4, arg5, arg6) ;            \
    GB_WHERE_C_LOGGER (C)

#define GB_WHERE5(C,arg2,arg3,arg4,arg5,where_string)               \
    GB_WHERE0 (where_string)                                        \
    info = GB_valid5 (C, arg2, arg3, arg4, arg5) ;                  \
    GB_WHERE_C_LOGGER (C)

#define GB_WHERE4(C,arg2,arg3,arg4,where_string)                    \
    GB_WHERE0 (where_string)                                        \
    info = GB_valid4 (C, arg2, arg3, arg4) ;                        \
    GB_WHERE_C_LOGGER (C)

#define GB_WHERE3(C,arg2,arg3,where_string)                         \
    GB_WHERE0 (where_string)                                        \
    info = GB_valid3 (C, arg2, arg3) ;                              \
    GB_WHERE_C_LOGGER (C)

#define GB_WHERE2(C,arg2,where_string)                              \
    GB_WHERE0 (where_string)                                        \
    info = GB_valid2 (C, arg2) ;                                    \
    GB_WHERE_C_LOGGER (C)

#define GB_WHERE1(C,where_string)                                   \
    GB_WHERE0 (where_string)                                        \
    info = GB_valid1 (C) ;                                          \
    GB_WHERE_C_LOGGER (C)

// GB_WHERE_n: check n arguments, no input/output matrix C for logger
#define GB_WHERE_1(arg1,where_string)                               \
    GB_WHERE0 (where_string)                                        \
    info = GB_valid1 (arg1) ;                                       \
    GB_WHERE_CHECK_INFO

#define GB_WHERE_2(arg1,arg2,where_string)                          \
    GB_WHERE0 (where_string)                                        \
    info = GB_valid2 (arg1, arg2) ;                                 \
    GB_WHERE_CHECK_INFO

#define GB_WHERE_3(arg1,arg2,arg3,where_string)                     \
    GB_WHERE0 (where_string)                                        \
    info = GB_valid3 (arg1, arg2, arg3) ;                           \
    GB_WHERE_CHECK_INFO

#define GB_WHERE_4(arg1,arg2,arg3,arg4,where_string)                \
    GB_WHERE0 (where_string)                                        \
    info = GB_valid4 (arg1, arg2, arg3, arg4) ;                     \
    GB_WHERE_CHECK_INFO

// for descriptors
#define GB_WHERE_DESC(desc,where_string)                            \
    GB_CHECK_INIT                                                   \
    GB_RETURN_IF_FAULTY (desc) ;                                    \
    GB_WERK (where_string)                                          \
    if (desc != NULL)                                               \
    {                                                               \
        /* free any prior error logged in the object */             \
        GB_FREE_MEMORY (&(desc->logger), desc->logger_size) ;       \
        Werk->logger_handle = &(desc->logger) ;                     \
        Werk->logger_size_handle = &(desc->logger_size) ;           \
    }

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
// The user can then retrieve the error string (owned by GraphBLAS) with:
//
//  const char *error ;
//  GrB_error (&error, A) ;

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
            (*logger_handle) = GB_CALLOC_MEMORY (GB_LOGGER_LEN+1,           \
                sizeof (char), logger_size_handle) ;                        \
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

// output cannot be readonly
#define GB_RETURN_IF_OUTPUT_IS_READONLY(arg)                            \
    if (GB_is_shallow ((GrB_Matrix) arg))                               \
    {                                                                   \
        return (GxB_OUTPUT_IS_READONLY) ;                               \
    }

// arg must be a matrix, vector, or scalar
#define GB_RETURN_IF_NULL_OR_INVALID(arg)                               \
    GB_RETURN_IF_NULL (arg) ;                                           \
    GB_RETURN_IF_INVALID (arg) ;

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
        return (C_replace ? GB_clear (C, Werk) : GrB_SUCCESS) ;             \
    }

//------------------------------------------------------------------------------
// GB_GET_DESCRIPTOR*: get the contents of a descriptor
//------------------------------------------------------------------------------

// check the descriptor and extract its contents
#define GB_GET_DESCRIPTOR(info,desc,dout,dmc,dms,d0,d1,dalgo,dsort)          \
    bool dout, dmc, dms, d0, d1 ;                                            \
    int dsort ;                                                              \
    int dalgo ;                                                              \
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

