//------------------------------------------------------------------------------
// GB_get_set.h: definitions for GrB_get/set methods
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_GET_SET_H
#define GB_GET_SET_H
#include "GB.h"

struct GB_Global_opaque
{
    int64_t magic ;
    size_t header_size ;
} ;

GrB_Type_Code GB_type_code_get  // return the GrB_Type_Code for the code
(
    const GB_Type_code code     // type code to convert
) ;

const char *GB_type_name_get (GrB_Type type) ;
const char *GB_code_name_get (GB_Type_code code, char *user_name) ;

GrB_Info GB_matvec_name_size_get (GrB_Matrix A, size_t *value, int field) ;
GrB_Info GB_matvec_name_get (GrB_Matrix A, char *name, int field) ;
GrB_Info GB_matvec_enum_get (GrB_Matrix A, int32_t *value, int field) ;
GrB_Info GB_matvec_name_set (GrB_Matrix A, char *value, int field) ;

GrB_Info GB_matvec_set
(
    GrB_Matrix A,
    bool is_vector,         // true if A is a GrB_Vector
    int32_t ivalue,
    double dvalue,
    int field,
    GB_Werk Werk
) ;

GrB_Info GB_op_enum_get   (GB_Operator op, int32_t *  value, GrB_Field field) ;
GrB_Info GB_op_scalar_get (GB_Operator op, GrB_Scalar value, GrB_Field field,
    GB_Werk Werk) ;
GrB_Info GB_op_string_get (GB_Operator op, char *     value, GrB_Field field) ;
GrB_Info GB_op_size_get   (GB_Operator op, size_t *   value, GrB_Field field) ;

const char *GB_op_name_get (GB_Operator op) ;
GrB_Info GB_op_string_set (GB_Operator op, char * value, GrB_Field field) ;

const char *GB_monoid_name_get (GrB_Monoid monoid) ;
const char *GB_semiring_name_get (GrB_Semiring semiring) ;

GrB_Info GB_op_or_type_string_set
(
    // input:
    bool user_defined,
    bool jitable,
    char *value,
    int field,
    // output:
    char **user_name,
    size_t *user_name_size,
    char *name,
    int32_t *name_len,
    char **defn,
    size_t *defn_size,
    uint64_t *hash
) ;

GrB_Info GB_monoid_get
(
    GrB_Monoid monoid,
    GrB_Scalar value,
    GrB_Field field,
    GB_Werk Werk
) ;

GrB_Info GB_user_name_set
(
    // input/output
    char **object_user_name,        // user_name of the object
    size_t *object_user_name_size,  // user_name_size of the object
    // input
    const char *new_name,           // new name for the object
    const bool only_once            // if true, the name of the object can
                                    // only be set once
) ;


#endif

