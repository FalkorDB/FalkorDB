/*
 * Copyright (c) 2019,2020 NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#ifndef GB_CONV_TYPE_H
#define GB_CONV_TYPE_H

extern "C" {
#include "GB.h"
};
#include <stdint.h>
#include <string>
#include <typeinfo>
#include <type_traits>
#include <memory>
#include <cstdlib>

/**---------------------------------------------------------------------------*
 * @file type_convert.hpp
 * @brief Defines the mapping between concrete C++ types and Grb types.
 *---------------------------------------------------------------------------**/
namespace cuda::jit {

template <typename T>
GrB_Type to_grb_type();

template<> inline GrB_Type to_grb_type<int8_t>() { return GrB_INT8; }
template<> inline GrB_Type to_grb_type<int16_t>() { return GrB_INT16; }
template<> inline GrB_Type to_grb_type<int32_t>() { return GrB_INT32; }
template<> inline GrB_Type to_grb_type<int64_t>() { return GrB_INT64; }
template<> inline GrB_Type to_grb_type<uint8_t>() { return GrB_UINT8; }
template<> inline GrB_Type to_grb_type<uint16_t>() { return GrB_UINT16; }
template<> inline GrB_Type to_grb_type<uint32_t>() { return GrB_UINT32; }
template<> inline GrB_Type to_grb_type<uint64_t>() { return GrB_UINT64; }
template<> inline GrB_Type to_grb_type<float>() { return GrB_FP32; }
template<> inline GrB_Type to_grb_type<double>() { return GrB_FP64; }
template<> inline GrB_Type to_grb_type<bool>() { return GrB_BOOL; }


template <typename T>
void set_element(GrB_Matrix A, T x, int64_t i, int64_t j);

template<> inline void set_element<int8_t>(GrB_Matrix A, int8_t x, int64_t i, int64_t j) { GrB_Matrix_setElement_INT8(A, x, i, j); }
template<> inline void set_element<int16_t>(GrB_Matrix A, int16_t x, int64_t i, int64_t j) { GrB_Matrix_setElement_INT16(A, x, i, j); }
template<> inline void set_element<int32_t>(GrB_Matrix A, int32_t x, int64_t i, int64_t j) { GrB_Matrix_setElement_INT32(A, x, i, j); }
template<> inline void set_element<int64_t>(GrB_Matrix A, int64_t x, int64_t i, int64_t j) { GrB_Matrix_setElement_INT64(A, x, i, j); }
template<> inline void set_element<uint8_t>(GrB_Matrix A, uint8_t x, int64_t i, int64_t j) { GrB_Matrix_setElement_UINT8(A, x, i, j); }
template<> inline void set_element<uint16_t>(GrB_Matrix A, uint16_t x, int64_t i, int64_t j) { GrB_Matrix_setElement_UINT16(A, x, i, j); }
template<> inline void set_element<uint32_t>(GrB_Matrix A, uint32_t x, int64_t i, int64_t j) { GrB_Matrix_setElement_UINT32(A, x, i, j); }
template<> inline void set_element<uint64_t>(GrB_Matrix A, uint64_t x, int64_t i, int64_t j) { GrB_Matrix_setElement_UINT64(A, x, i, j); }
template<> inline void set_element<float>(GrB_Matrix A, float x, int64_t i, int64_t j) { GrB_Matrix_setElement_FP32(A, x, i, j); }
template<> inline void set_element<double>(GrB_Matrix A, double x, int64_t i, int64_t j) { GrB_Matrix_setElement_FP64(A, x, i, j); }
template<> inline void set_element<bool>(GrB_Matrix A, bool x, int64_t i, int64_t j) { GrB_Matrix_setElement_BOOL(A, x, i, j); }


template <typename T>
void vector_set_element(GrB_Vector A, T x, int64_t i);

template<> inline void vector_set_element<int8_t>(GrB_Vector A, int8_t x, int64_t i) { GrB_Vector_setElement_INT8(A, x, i); }
template<> inline void vector_set_element<int16_t>(GrB_Vector A, int16_t x, int64_t i) { GrB_Vector_setElement_INT16(A, x, i); }
template<> inline void vector_set_element<int32_t>(GrB_Vector A, int32_t x, int64_t i) { GrB_Vector_setElement_INT32(A, x, i); }
template<> inline void vector_set_element<int64_t>(GrB_Vector A, int64_t x, int64_t i) { GrB_Vector_setElement_INT64(A, x, i); }
template<> inline void vector_set_element<uint8_t>(GrB_Vector A, uint8_t x, int64_t i) { GrB_Vector_setElement_UINT8(A, x, i); }
template<> inline void vector_set_element<uint16_t>(GrB_Vector A, uint16_t x, int64_t i) { GrB_Vector_setElement_UINT16(A, x, i); }
template<> inline void vector_set_element<uint32_t>(GrB_Vector A, uint32_t x, int64_t i) { GrB_Vector_setElement_UINT32(A, x, i); }
template<> inline void vector_set_element<uint64_t>(GrB_Vector A, uint64_t x, int64_t i) { GrB_Vector_setElement_UINT64(A, x, i); }
template<> inline void vector_set_element<float>(GrB_Vector A, float x, int64_t i) { GrB_Vector_setElement_FP32(A, x, i); }
template<> inline void vector_set_element<double>(GrB_Vector A, double x, int64_t i) { GrB_Vector_setElement_FP64(A, x, i); }
template<> inline void vector_set_element<bool>(GrB_Vector A, bool x, int64_t i) { GrB_Vector_setElement_BOOL(A, x, i); }


    template <typename T>
    void scalar_set_element(GrB_Scalar A, T x);

    template<> inline void scalar_set_element<int8_t>(GrB_Scalar A, int8_t x) { GrB_Scalar_setElement_INT8(A, x); }
    template<> inline void scalar_set_element<int16_t>(GrB_Scalar A, int16_t x) { GrB_Scalar_setElement_INT16(A, x); }
    template<> inline void scalar_set_element<int32_t>(GrB_Scalar A, int32_t x) { GrB_Scalar_setElement_INT32(A, x); }
    template<> inline void scalar_set_element<int64_t>(GrB_Scalar A, int64_t x) { GrB_Scalar_setElement_INT64(A, x); }
    template<> inline void scalar_set_element<uint8_t>(GrB_Scalar A, uint8_t x) { GrB_Scalar_setElement_UINT8(A, x); }
    template<> inline void scalar_set_element<uint16_t>(GrB_Scalar A, uint16_t x) { GrB_Scalar_setElement_UINT16(A, x); }
    template<> inline void scalar_set_element<uint32_t>(GrB_Scalar A, uint32_t x) { GrB_Scalar_setElement_UINT32(A, x); }
    template<> inline void scalar_set_element<uint64_t>(GrB_Scalar A, uint64_t x) { GrB_Scalar_setElement_UINT64(A, x); }
    template<> inline void scalar_set_element<float>(GrB_Scalar A, float x) { GrB_Scalar_setElement_FP32(A, x); }
    template<> inline void scalar_set_element<double>(GrB_Scalar A, double x) { GrB_Scalar_setElement_FP64(A, x); }
    template<> inline void scalar_set_element<bool>(GrB_Scalar A, bool x) { GrB_Scalar_setElement_BOOL(A, x); }


template<typename T>
GrB_Info vector_reduce(T *scalar, GrB_Vector A, GrB_Monoid op);

template<> inline GrB_Info vector_reduce<int8_t>(int8_t *scalar, GrB_Vector A, GrB_Monoid op) { return GrB_Vector_reduce_INT8(scalar, NULL, op, A, NULL); }
template<> inline GrB_Info vector_reduce<int16_t>(int16_t *scalar, GrB_Vector A, GrB_Monoid op) { return GrB_Vector_reduce_INT16(scalar, NULL, op, A, NULL); }
template<> inline GrB_Info vector_reduce<int32_t>(int32_t *scalar, GrB_Vector A, GrB_Monoid op) { return GrB_Vector_reduce_INT32(scalar, NULL, op, A, NULL); }
template<> inline GrB_Info vector_reduce<int64_t>(int64_t *scalar, GrB_Vector A, GrB_Monoid op) { return GrB_Vector_reduce_INT64(scalar, NULL, op, A, NULL); }
template<> inline GrB_Info vector_reduce<uint8_t>(uint8_t *scalar, GrB_Vector A, GrB_Monoid op) { return GrB_Vector_reduce_UINT8(scalar, NULL, op, A, NULL); }
template<> inline GrB_Info vector_reduce<uint16_t>(uint16_t *scalar, GrB_Vector A, GrB_Monoid op) { return GrB_Vector_reduce_UINT16(scalar, NULL, op, A, NULL); }
template<> inline GrB_Info vector_reduce<uint32_t>(uint32_t *scalar, GrB_Vector A, GrB_Monoid op) { return GrB_Vector_reduce_UINT32(scalar, NULL, op, A, NULL); }
template<> inline GrB_Info vector_reduce<uint64_t>(uint64_t *scalar, GrB_Vector A, GrB_Monoid op) { return GrB_Vector_reduce_UINT64(scalar, NULL, op, A, NULL); }
template<> inline GrB_Info vector_reduce<float>(float *scalar, GrB_Vector A, GrB_Monoid op) { return GrB_Vector_reduce_FP32(scalar, NULL, op, A, NULL); }
template<> inline GrB_Info vector_reduce<double>(double *scalar, GrB_Vector A, GrB_Monoid op) { return GrB_Vector_reduce_FP64(scalar, NULL, op, A, NULL); }
template<> inline GrB_Info vector_reduce<bool>(bool *scalar, GrB_Vector A, GrB_Monoid op) { return GrB_Vector_reduce_BOOL(scalar, NULL, op, A, NULL); }

/**
 *     GxB_Matrix_reduce_FC32     // c = accum (c, reduce_to_scalar (A))
            (
                    GxB_FC32_t *c,                  // result scalar
    const GrB_BinaryOp accum,       // optional accum for c=accum(c,t)
    const GrB_Monoid monoid,        // monoid to do the reduction
    const GrB_Matrix A,             // matrix to reduce
    const GrB_Descriptor desc

 * @tparam T 
 * @param scalar 
 * @param A 
 * @param op 
 * @return 
 */

template<typename T>
GrB_Info matrix_reduce(T *scalar, GrB_Matrix A, GrB_Monoid op);

template<> inline GrB_Info matrix_reduce<int8_t>(int8_t *scalar, GrB_Matrix A, GrB_Monoid op) { return GrB_Matrix_reduce_INT8(scalar, NULL, op, A, NULL); }
template<> inline GrB_Info matrix_reduce<int16_t>(int16_t *scalar, GrB_Matrix A, GrB_Monoid op) { return GrB_Matrix_reduce_INT16(scalar, NULL, op, A, NULL); }
template<> inline GrB_Info matrix_reduce<int32_t>(int32_t *scalar, GrB_Matrix A, GrB_Monoid op) { return GrB_Matrix_reduce_INT32(scalar, NULL, op, A, NULL); }
template<> inline GrB_Info matrix_reduce<int64_t>(int64_t *scalar, GrB_Matrix A, GrB_Monoid op) { return GrB_Matrix_reduce_INT64(scalar, NULL, op, A, NULL); }
template<> inline GrB_Info matrix_reduce<uint8_t>(uint8_t *scalar, GrB_Matrix A, GrB_Monoid op) { return GrB_Matrix_reduce_UINT8(scalar, NULL, op, A, NULL); }
template<> inline GrB_Info matrix_reduce<uint16_t>(uint16_t *scalar, GrB_Matrix A, GrB_Monoid op) { return GrB_Matrix_reduce_UINT16(scalar, NULL, op, A, NULL); }
template<> inline GrB_Info matrix_reduce<uint32_t>(uint32_t *scalar, GrB_Matrix A, GrB_Monoid op) { return GrB_Matrix_reduce_UINT32(scalar, NULL, op, A, NULL); }
template<> inline GrB_Info matrix_reduce<uint64_t>(uint64_t *scalar, GrB_Matrix A, GrB_Monoid op) { return GrB_Matrix_reduce_UINT64(scalar, NULL, op, A, NULL); }
template<> inline GrB_Info matrix_reduce<float>(float *scalar, GrB_Matrix A, GrB_Monoid op) { return GrB_Matrix_reduce_FP32(scalar, NULL, op, A, NULL); }
template<> inline GrB_Info matrix_reduce<double>(double *scalar, GrB_Matrix A, GrB_Monoid op) { return GrB_Matrix_reduce_FP64(scalar, NULL, op, A, NULL); }
template<> inline GrB_Info matrix_reduce<bool>(bool *scalar, GrB_Matrix A, GrB_Monoid op) { return GrB_Matrix_reduce_BOOL(scalar, NULL, op, A, NULL); }


template <typename T>
GrB_Info get_element(GrB_Matrix A, T* x, int64_t i, int64_t j);
template<> inline GrB_Info get_element<int8_t>(GrB_Matrix A, int8_t *x, int64_t i, int64_t j) { return GrB_Matrix_extractElement_INT8(x, A, i, j); }
template<> inline GrB_Info get_element<int16_t>(GrB_Matrix A, int16_t *x, int64_t i, int64_t j) { return GrB_Matrix_extractElement_INT16(x, A, i, j); }
template<> inline GrB_Info get_element<int32_t>(GrB_Matrix A, int32_t *x, int64_t i, int64_t j) { return GrB_Matrix_extractElement_INT32(x, A, i, j); }
template<> inline GrB_Info get_element<int64_t>(GrB_Matrix A, int64_t *x, int64_t i, int64_t j) { return GrB_Matrix_extractElement_INT64(x, A, i, j); }
template<> inline GrB_Info get_element<uint8_t>(GrB_Matrix A, uint8_t *x, int64_t i, int64_t j) { return GrB_Matrix_extractElement_UINT8(x, A, i, j); }
template<> inline GrB_Info get_element<uint16_t>(GrB_Matrix A, uint16_t *x, int64_t i, int64_t j) { return GrB_Matrix_extractElement_UINT16(x, A, i, j); }
template<> inline GrB_Info get_element<uint32_t>(GrB_Matrix A, uint32_t *x, int64_t i, int64_t j) { return GrB_Matrix_extractElement_UINT32(x, A, i, j); }
template<> inline GrB_Info get_element<uint64_t>(GrB_Matrix A, uint64_t *x, int64_t i, int64_t j) { return GrB_Matrix_extractElement_UINT64(x, A, i, j); }
template<> inline GrB_Info get_element<float>(GrB_Matrix A, float *x, int64_t i, int64_t j) { return GrB_Matrix_extractElement_FP32(x, A, i, j); }
template<> inline GrB_Info get_element<double>(GrB_Matrix A, double *x, int64_t i, int64_t j) { return GrB_Matrix_extractElement_FP64(x, A, i, j); }
template<> inline GrB_Info get_element<bool>(GrB_Matrix A, bool *x, int64_t i, int64_t j) { return GrB_Matrix_extractElement_BOOL(x, A, i, j); }





template<typename T>
class type_name {
public:
    static const char *name;
};

#define DECLARE_TYPE_NAME(x)  template<> inline const char *type_name<x>::name = #x;
#define GET_TYPE_NAME(x) (type_name<decltype(x)>::name)

    DECLARE_TYPE_NAME(int);
    DECLARE_TYPE_NAME(int&);
    DECLARE_TYPE_NAME(int*);
    DECLARE_TYPE_NAME(int8_t);
    DECLARE_TYPE_NAME(int8_t&);
    DECLARE_TYPE_NAME(int8_t*);
    DECLARE_TYPE_NAME(unsigned char);
    DECLARE_TYPE_NAME(unsigned char&);
    DECLARE_TYPE_NAME(unsigned char*);
//  DECLARE_TYPE_NAME(unsigned int);
//  DECLARE_TYPE_NAME(unsigned int&);
//  DECLARE_TYPE_NAME(unsigned int*);
//  DECLARE_TYPE_NAME(unsigned int32_t);
//  DECLARE_TYPE_NAME(unsigned int32_t&);
//  DECLARE_TYPE_NAME(unsigned int32_t*);
    DECLARE_TYPE_NAME(unsigned int64_t);
    DECLARE_TYPE_NAME(unsigned int64_t&);
    DECLARE_TYPE_NAME(unsigned int64_t*);
    DECLARE_TYPE_NAME(long);
    DECLARE_TYPE_NAME(long&);
    DECLARE_TYPE_NAME(long*);
    DECLARE_TYPE_NAME(float);
    DECLARE_TYPE_NAME(float&);
    DECLARE_TYPE_NAME(float*);
    DECLARE_TYPE_NAME(double);
    DECLARE_TYPE_NAME(double&);
    DECLARE_TYPE_NAME(double*);
    DECLARE_TYPE_NAME(bool);



    inline const std::string grb_str_type(GB_Type_code grb_type_code) {
        switch(grb_type_code) {
            case GB_BOOL_code:
                return "bool";
            case GB_INT8_code:
                return "int8_t";
            case GB_UINT8_code:
                return "uint8_t";
            case GB_INT16_code:
                return "int16_t";
            case GB_UINT16_code:
                return "uint16_t";
            case GB_INT32_code:
                return "int32_t";
            case GB_UINT32_code:
                return "uint32_t";
            case GB_INT64_code:
                return "int64_t";
            case GB_UINT64_code:
                return "uint64_t";
            case GB_FP32_code:
                return "float";
            case GB_FP64_code:
                return "double";
            default:
                printf("Error: GrB_Type not supported.\n");
                exit(1);
        }
    }


}  // namespace cuda::jit
#endif
