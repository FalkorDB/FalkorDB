//------------------------------------------------------------------------------
// GB_stringify.h: prototype definitions construction of *.h definitions
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_STRINGIFY_H
#define GB_STRINGIFY_H

#include "GB_binop.h"
#include "GB_jitifyer.h"
#include "GB_callback.h"

//------------------------------------------------------------------------------
// print kernel preface
//------------------------------------------------------------------------------

void GB_macrofy_preface
(
    FILE *fp,               // target file to write, already open
    char *kernel_name,      // name of the kernel
    char *preface           // user-provided preface
) ;

//------------------------------------------------------------------------------
// left and right shift
//------------------------------------------------------------------------------

#define GB_LSHIFT(x,k) (((uint64_t) x) << k)
#define GB_RSHIFT(x,k,b) ((x >> k) & ((((uint64_t)0x00000001) << b) -1))

//------------------------------------------------------------------------------
// GB_macrofy_name: create the kernel name
//------------------------------------------------------------------------------

#define GB_KLEN (100 + 2*GxB_MAX_NAME_LEN)

void GB_macrofy_name
(
    // output:
    char *kernel_name,      // string of length GB_KLEN
    // input
    const char *name_space, // namespace for the kernel_name
    const char *kname,      // kname for the kernel_name
    int scode_digits,       // # of hexadecimal digits printed
    uint64_t scode,         // enumify'd code of the kernel
    const char *suffix      // suffix for the kernel_name (NULL if none)
) ;

GrB_Info GB_demacrofy_name
(
    // input/output:
    char *kernel_name,      // string of length GB_KLEN; NUL's are inserted
                            // to demarcate each part of the kernel_name.
    // output
    char **name_space,      // namespace for the kernel_name
    char **kname,           // kname for the kernel_name
    uint64_t *scode,        // enumify'd code of the kernel
    char **suffix           // suffix for the kernel_name (NULL if none)
) ;

//------------------------------------------------------------------------------
// GrB_reduce
//------------------------------------------------------------------------------

uint64_t GB_encodify_reduce // encode a GrB_reduce problem
(
    // output:
    GB_jit_encoding *encoding,  // unique encoding of the entire problem,
                                // except for the suffix
    char **suffix,              // suffix for user-defined kernel
    // input:
    GrB_Monoid monoid,      // the monoid to enumify
    GrB_Matrix A            // input matrix to reduce
) ;

void GB_enumify_reduce      // enumerate a GrB_reduce problem
(
    // output:
    uint64_t *rcode,        // unique encoding of the entire problem
    // input:
    GrB_Monoid monoid,      // the monoid to enumify
    GrB_Matrix A            // input matrix to monoid
) ;

void GB_macrofy_reduce      // construct all macros for GrB_reduce to scalar
(
    FILE *fp,               // target file to write, already open
    // input:
    uint64_t rcode,         // encoded problem
    GrB_Monoid monoid,      // monoid to macrofy
    GrB_Type atype          // type of the A matrix to reduce
) ;

GrB_Info GB_reduce_to_scalar_jit    // z = reduce_to_scalar (A) via the JIT
(
    // output:
    void *z,                    // result
    // input:
    const GrB_Monoid monoid,    // monoid to do the reduction
    const GrB_Matrix A,         // matrix to reduce
    GB_void *restrict W,        // workspace
    bool *restrict F,           // workspace
    int ntasks,                 // # of tasks to use
    int nthreads                // # of threads to use
) ;

//------------------------------------------------------------------------------
// GrB_eWiseAdd, GrB_eWiseMult, GxB_eWiseUnion
//------------------------------------------------------------------------------

// FUTURE: add accumulator for eWise operations?

uint64_t GB_encodify_ewise      // encode an ewise problem
(
    // output:
    GB_jit_encoding *encoding,  // unique encoding of the entire problem,
                                // except for the suffix
    char **suffix,              // suffix for user-defined kernel
    // input:
    const GB_jit_kcode kcode,   // kernel to encode
    const bool is_eWiseMult,    // if true, method is emult
    const bool C_iso,
    const bool C_in_iso,
    const int C_sparsity,
    const GrB_Type ctype,
    const GrB_Matrix M,
    const bool Mask_struct,
    const bool Mask_comp,
    const GrB_BinaryOp binaryop,
    const bool flipxy,
    const GrB_Matrix A,
    const GrB_Matrix B
) ;

void GB_enumify_ewise       // enumerate a GrB_eWise problem
(
    // output:
    uint64_t *scode,        // unique encoding of the entire operation
    // input:
    bool is_eWiseMult,      // if true, method is emult
    bool is_eWiseUnion,     // if true, method is eWiseUnion
    bool can_copy_to_C,     // if true C(i,j)=A(i,j) can bypass the op
    // C matrix:
    bool C_iso,             // if true, C is iso on output
    bool C_in_iso,          // if true, C is iso on input
    int C_sparsity,         // sparse, hyper, bitmap, or full
    GrB_Type ctype,         // C=((ctype) T) is the final typecast
    // M matrix:
    GrB_Matrix M,           // may be NULL
    bool Mask_struct,       // mask is structural
    bool Mask_comp,         // mask is complemented
    // operator:
    GrB_BinaryOp binaryop,  // the binary operator to enumify
    bool flipxy,            // multiplier is: op(a,b) or op(b,a)
    // A and B:
    GrB_Matrix A,           // NULL for unary apply with binop, bind 1st
    GrB_Matrix B            // NULL for unary apply with binop, bind 2nd
) ;

void GB_macrofy_ewise           // construct all macros for GrB_eWise
(
    // output:
    FILE *fp,                   // target file to write, already open
    // input:
    uint64_t scode,
    GrB_BinaryOp binaryop,      // binaryop to macrofy
    GrB_Type ctype,
    GrB_Type atype,
    GrB_Type btype
) ;

GrB_Info GB_add_jit      // C=A+B, C<#M>=A+B, add, via the JIT
(
    // input/output:
    GrB_Matrix C,
    // input:
    const int C_sparsity,
    const GrB_Matrix M,
    const bool Mask_struct,
    const bool Mask_comp,
    const GrB_BinaryOp binaryop,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const bool Ch_is_Mh,
    const int64_t *restrict C_to_M,
    const int64_t *restrict C_to_A,
    const int64_t *restrict C_to_B,
    const GB_task_struct *restrict TaskList,
    const int C_ntasks,
    const int C_nthreads,
    const int64_t *restrict M_ek_slicing,
    const int M_nthreads,
    const int M_ntasks,
    const int64_t *restrict A_ek_slicing,
    const int A_nthreads,
    const int A_ntasks,
    const int64_t *restrict B_ek_slicing,
    const int B_nthreads,
    const int B_ntasks
) ;

GrB_Info GB_union_jit      // C=A+B, C<#M>=A+B, eWiseUnion, via the JIT
(
    // input/output:
    GrB_Matrix C,
    // input:
    const int C_sparsity,
    const GrB_Matrix M,
    const bool Mask_struct,
    const bool Mask_comp,
    const GrB_BinaryOp binaryop,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const GB_void *alpha_scalar_in,
    const GB_void *beta_scalar_in,
    const bool Ch_is_Mh,
    const int64_t *restrict C_to_M,
    const int64_t *restrict C_to_A,
    const int64_t *restrict C_to_B,
    const GB_task_struct *restrict TaskList,
    const int C_ntasks,
    const int C_nthreads,
    const int64_t *restrict M_ek_slicing,
    const int M_nthreads,
    const int M_ntasks,
    const int64_t *restrict A_ek_slicing,
    const int A_nthreads,
    const int A_ntasks,
    const int64_t *restrict B_ek_slicing,
    const int B_nthreads,
    const int B_ntasks
) ;

GrB_Info GB_emult_08_jit      // C<#M>=A.*B, emult_08, via the JIT
(
    // input/output:
    GrB_Matrix C,
    // input:
    const int C_sparsity,
    const GrB_Matrix M,
    const bool Mask_struct,
    const bool Mask_comp,
    const GrB_BinaryOp binaryop,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const int64_t *restrict C_to_M,
    const int64_t *restrict C_to_A,
    const int64_t *restrict C_to_B,
    const GB_task_struct *restrict TaskList,
    const int C_ntasks,
    const int C_nthreads
) ;

GrB_Info GB_emult_02_jit      // C<#M>=A.*B, emult_02, via the JIT
(
    // input/output:
    GrB_Matrix C,
    // input:
    const int C_sparsity,
    const GrB_Matrix M,
    const bool Mask_struct,
    const bool Mask_comp,
    const GrB_BinaryOp binaryop,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const int64_t *restrict Cp_kfirst,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB_emult_03_jit      // C<#M>=A.*B, emult_03, via the JIT
(
    // input/output:
    GrB_Matrix C,
    // input:
    const int C_sparsity,
    const GrB_Matrix M,
    const bool Mask_struct,
    const bool Mask_comp,
    const GrB_BinaryOp binaryop,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const int64_t *restrict Cp_kfirst,
    const int64_t *B_ek_slicing,
    const int B_ntasks,
    const int B_nthreads
) ;

GrB_Info GB_emult_04_jit      // C<M>=A.*B, emult_04, via the JIT
(
    // input/output:
    GrB_Matrix C,
    // input:
    const int C_sparsity,
    const GrB_Matrix M,
    const bool Mask_struct,
    const GrB_BinaryOp binaryop,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const int64_t *restrict Cp_kfirst,
    const int64_t *M_ek_slicing,
    const int M_ntasks,
    const int M_nthreads
) ;

GrB_Info GB_emult_bitmap_jit      // C<#M>=A.*B, emult_bitmap, via the JIT
(
    // input/output:
    GrB_Matrix C,
    // input:
    const GrB_Matrix M,
    const bool Mask_struct,
    const bool Mask_comp,
    const GrB_BinaryOp binaryop,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const int64_t *M_ek_slicing,
    const int M_ntasks,
    const int M_nthreads,
    const int C_nthreads
) ;

GrB_Info GB_ewise_fulla_jit    // C+=A+B via the JIT
(
    // input/output:
    GrB_Matrix C,
    // input:
    const GrB_BinaryOp binaryop,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const int nthreads
) ;

GrB_Info GB_ewise_fulln_jit  // C=A+B via the JIT
(
    // input/output:
    GrB_Matrix C,
    // input:
    const GrB_BinaryOp binaryop,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const int nthreads
) ;

GrB_Info GB_rowscale_jit      // C=D*B, rowscale, via the JIT
(
    // input/output:
    GrB_Matrix C,
    // input:
    const GrB_Matrix D,
    const GrB_Matrix B,
    const GrB_BinaryOp binaryop,
    const bool flipxy,
    const int nthreads
) ;

GrB_Info GB_colscale_jit      // C=A*D, colscale, via the JIT
(
    // input/output:
    GrB_Matrix C,
    // input:
    const GrB_Matrix A,
    const GrB_Matrix D,
    const GrB_BinaryOp binaryop,
    const bool flipxy,
    const int64_t *restrict A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

//------------------------------------------------------------------------------
// GrB_mxm
//------------------------------------------------------------------------------

// FUTURE: add accumulator for mxm?

uint64_t GB_encodify_mxm        // encode a GrB_mxm problem
(
    // output:
    GB_jit_encoding *encoding,  // unique encoding of the entire problem,
                                // except for the suffix
    char **suffix,              // suffix for user-defined kernel
    // input:
    const GB_jit_kcode kcode,   // kernel to encode
    const bool C_iso,
    const bool C_in_iso,
    const int C_sparsity,
    const GrB_Type ctype,
    const GrB_Matrix M,
    const bool Mask_struct,
    const bool Mask_comp,
    const GrB_Semiring semiring,
    const bool flipxy,
    const GrB_Matrix A,
    const GrB_Matrix B
) ;

void GB_enumify_mxm         // enumerate a GrB_mxm problem
(
    // output:              // future:: may need to become 2 x uint64
    uint64_t *scode,        // unique encoding of the entire semiring
    // input:
    // C matrix:
    bool C_iso,             // C output iso: if true, semiring is ANY_PAIR_BOOL
    bool C_in_iso,          // C input iso status
    int C_sparsity,         // sparse, hyper, bitmap, or full
    GrB_Type ctype,         // C=((ctype) T) is the final typecast
    // M matrix:
    GrB_Matrix M,           // may be NULL
    bool Mask_struct,       // mask is structural
    bool Mask_comp,         // mask is complemented
    // semiring:
    GrB_Semiring semiring,  // the semiring to enumify
    bool flipxy,            // multiplier is: mult(a,b) or mult(b,a)
    // A and B:
    GrB_Matrix A,
    GrB_Matrix B
) ;

void GB_macrofy_mxm        // construct all macros for GrB_mxm
(
    // output:
    FILE *fp,                   // target file to write, already open
    // input:
    uint64_t scode,
    GrB_Semiring semiring,  // the semiring to macrofy
    GrB_Type ctype,
    GrB_Type atype,
    GrB_Type btype
) ;

GrB_Info GB_AxB_saxpy3_jit      // C<M>=A*B, saxpy3, via the JIT
(
    // input/output:
    GrB_Matrix C,
    // input:
    const GrB_Matrix M,
    const bool Mask_comp,
    const bool Mask_struct,
    const bool M_in_place,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const GrB_Semiring semiring,
    const bool flipxy,
    void *SaxpyTasks,
    const int ntasks,
    const int nfine,
    const int nthreads,
    const int do_sort,          // if nonzero, try to sort in saxpy3
    GB_Werk Werk
) ;

GrB_Info GB_AxB_saxpy4_jit          // C+=A*B, saxpy4 method, via the JIT
(
    // input/output:
    GrB_Matrix C,
    // input:
    const GrB_Matrix A,
    const GrB_Matrix B,
    const GrB_Semiring semiring,
    const bool flipxy,
    const int ntasks,
    const int nthreads,
    const int nfine_tasks_per_vector,
    const bool use_coarse_tasks,
    const bool use_atomics,
    const int64_t *A_slice,
    const int64_t *H_slice,
    GB_void *restrict Wcx,
    int8_t *restrict Wf
) ;

GrB_Info GB_AxB_saxpy5_jit          // C+=A*B, saxpy5 method, via the JIT
(
    // input/output:
    GrB_Matrix C,
    // input:
    const GrB_Matrix A,
    const GrB_Matrix B,
    const GrB_Semiring semiring,
    const bool flipxy,
    const int ntasks,
    const int nthreads,
    const int64_t *B_slice
) ;

GrB_Info GB_AxB_saxbit_jit      // C<M>=A*B, saxbit, via the JIT
(
    // input/output:
    GrB_Matrix C,
    // input:
    const GrB_Matrix M,
    const bool Mask_comp,
    const bool Mask_struct,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const GrB_Semiring semiring,
    const bool flipxy,
    const int ntasks,
    const int nthreads,
    const int nfine_tasks_per_vector,
    const bool use_coarse_tasks,
    const bool use_atomics,
    const int64_t *restrict M_ek_slicing,
    const int M_nthreads,
    const int M_ntasks,
    const int64_t *restrict A_slice,
    const int64_t *restrict H_slice,
    GB_void *restrict Wcx,
    int8_t *restrict Wf
) ;

GrB_Info GB_AxB_dot2_jit        // C<M>=A'*B, dot2 method, via the JIT
(
    // input/output:
    GrB_Matrix C,
    // input:
    const GrB_Matrix M,
    const bool Mask_comp,
    const bool Mask_struct,
    const GrB_Matrix A,
    const int64_t *restrict A_slice,
    const GrB_Matrix B,
    const int64_t *restrict B_slice,
    const GrB_Semiring semiring,
    const bool flipxy,
    const int nthreads,
    const int naslice,
    const int nbslice
) ;

GrB_Info GB_AxB_dot2n_jit        // C<M>=A*B, dot2n method, via the JIT
(
    // input/output:
    GrB_Matrix C,
    // input:
    const GrB_Matrix M,
    const bool Mask_comp,
    const bool Mask_struct,
    const GrB_Matrix A,
    const int64_t *restrict A_slice,
    const GrB_Matrix B,
    const int64_t *restrict B_slice,
    const GrB_Semiring semiring,
    const bool flipxy,
    const int nthreads,
    const int naslice,
    const int nbslice
) ;

GrB_Info GB_AxB_dot3_jit        // C<M>=A'B, dot3, via the JIT
(
    // input/output:
    GrB_Matrix C,               // never iso for this kernel
    // input:
    const GrB_Matrix M, const bool Mask_struct,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const GrB_Semiring semiring,
    const bool flipxy,
    const GB_task_struct *restrict TaskList,
    const int ntasks,
    const int nthreads
) ;

GrB_Info GB_AxB_dot4_jit            // C+=A'*B, dot4 method, via the JIT
(
    // input/output:
    GrB_Matrix C,
    // input:
    const bool C_in_iso,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const GrB_Semiring semiring,
    const bool flipxy,
    const int64_t *restrict A_slice,
    const int64_t *restrict B_slice,
    const int naslice,
    const int nbslice,
    const int nthreads,
    GB_Werk Werk
) ;

//------------------------------------------------------------------------------
// enumify and macrofy the mask matrix M
//------------------------------------------------------------------------------

void GB_enumify_mask       // return enum to define mask macros
(
    // output:
    int *mask_ecode,            // enumified mask
    // input
    const GB_Type_code mcode,   // typecode of the mask matrix M,
                                // or 0 if M is not present
    bool Mask_struct,           // true if M structural, false if valued
    bool Mask_comp              // true if M complemented
) ;

void GB_macrofy_mask
(
    FILE *fp,               // file to write macros, assumed open already
    // input:
    int mask_ecode,         // enumified mask
    char *Mname,            // name of the mask
    int msparsity           // sparsity of the mask
) ;

//------------------------------------------------------------------------------
// enumify and macrofy a monoid
//------------------------------------------------------------------------------

void GB_enumify_monoid  // enumerate a monoid
(
    // outputs:
    int *add_ecode,     // binary op as an enum
    int *id_ecode,      // identity value as an enum
    int *term_ecode,    // terminal value as an enum
    // inputs:
    int add_opcode,     // must be a built-in binary operator from a monoid
    int zcode           // type of the monoid (x, y, and z)
) ;

void GB_macrofy_monoid  // construct the macros for a monoid
(
    FILE *fp,           // File to write macros, assumed open already
    // inputs:
    int add_ecode,      // binary op as an enum
    int id_ecode,       // identity value as an enum
    int term_ecode,     // terminal value as an enum (<= 28 is terminal)
    bool C_iso,         // true if C is iso
    GrB_Monoid monoid,  // monoid to macrofy
    bool disable_terminal_condition,    // if true, the monoid is assumed
                        // to be non-terminal.  For the (times, firstj, int64)
                        // semiring, times is normally a terminal monoid, but
                        // it's not worth exploiting in GrB_mxm.
    // output:
    const char **u_expression
) ;

bool GB_enumify_cuda_atomic         // return true if CUDA can do it atomically
(
    // output:
    const char **a,                 // CUDA atomic function name
    bool *user_monoid_atomically,   // true if user monoid has an atomic update
    const char **cuda_type,         // CUDA atomic type
    // input:
    GrB_Monoid monoid,  // monoid to query
    int add_ecode,      // binary op as an enum
    size_t zsize,       // ztype->size
    int zcode           // ztype->code
) ;

void GB_macrofy_query
(
    FILE *fp,
    const bool builtin, // true if method is all builtin
    GrB_Monoid monoid,  // monoid for reduce or semiring; NULL otherwise
    GB_Operator op0,    // monoid op, select op, unary op, etc
    GB_Operator op1,    // binaryop for a semring
    GrB_Type type0,
    GrB_Type type1,
    GrB_Type type2,
    uint64_t hash       // hash code for the kernel
) ;

//------------------------------------------------------------------------------
// binary operators
//------------------------------------------------------------------------------

void GB_enumify_binop
(
    // output:
    int *ecode,         // enumerated operator, range 0 to 110; -1 on failure
    // input:
    GB_Opcode opcode,   // opcode of GraphBLAS operator to convert into a macro
    GB_Type_code zcode, // op->xtype->code of the operator
    bool for_semiring   // true for A*B, false for A+B or A.*B
) ;

void GB_macrofy_binop
(
    FILE *fp,
    // input:
    const char *macro_name,
    bool flipxy,                // if true: op is f(y,x)
    bool is_monoid_or_build,    // if true: additive operator for monoid,
                                // or binary op for GrB_Matrix_build
    bool is_ewise,              // if true: binop for ewise methods
    int ecode,
    bool C_iso,                 // if true: C is iso
    GrB_BinaryOp op,            // NULL if C is iso
    // output:
    const char **f_handle,
    const char **u_handle
) ;

//------------------------------------------------------------------------------
// operator definitions and typecasting
//------------------------------------------------------------------------------

void GB_macrofy_defn    // construct a defn for an operator
(
    FILE *fp,
    int kind,           // 0: built-in function
                        // 1: built-in macro
                        // 2: built-in macro needed for CUDA only
                        // 3: user-defined function or macro
    const char *name,
    const char *defn
) ;

void GB_macrofy_string
(
    FILE *fp,
    const char *name,
    const char *defn
) ;

const char *GB_macrofy_cast_expression  // return cast expression
(
    FILE *fp,
    // input:
    GrB_Type ztype,     // output type
    GrB_Type xtype,     // input type
    // output
    int *nargs          // # of string arguments in output format
) ;

void GB_macrofy_cast_input
(
    FILE *fp,
    // input:
    const char *macro_name,     // name of the macro: #define macro(z,x...)
    const char *zarg,           // name of the z argument of the macro
    const char *xargs,          // one or more x arguments
    const char *xexpr,          // an expression based on xargs
    const GrB_Type ztype,       // the type of the z output
    const GrB_Type xtype        // the type of the x input
) ;

void GB_macrofy_cast_output
(
    FILE *fp,
    // input:
    const char *macro_name,     // name of the macro: #define macro(z,x...)
    const char *zarg,           // name of the z argument of the macro
    const char *xargs,          // one or more x arguments
    const char *xexpr,          // an expression based on xargs
    const GrB_Type ztype,       // the type of the z input
    const GrB_Type xtype        // the type of the x output
) ;

void GB_macrofy_cast_copy
(
    FILE *fp,
    // input:
    const char *cname,          // name of the C matrix (typically "C")
    const char *aname,          // name of the A matrix (typically "A" or "B")
    const GrB_Type ctype,       // the type of the C matrix
    const GrB_Type atype,       // the type of the A matrix
    const bool A_iso            // true if A is iso
) ;

void GB_macrofy_input
(
    FILE *fp,
    // input:
    const char *aname,      // name of the scalar aij = ...
    const char *Amacro,     // name of the macro is GB_GET*(Amacro)
    const char *Aname,      // name of the input matrix
    bool do_matrix_macros,  // if true, do the matrix macros
    GrB_Type xtype,         // type of aij
    GrB_Type atype,         // type of the input matrix
    int asparsity,          // sparsity format of the input matrix
    int acode,              // type code of the input (0 if pattern)
    int A_iso_code,         // 1 if A is iso
    int azombies            // 1 if A has zombies, 0 if A has no zombies,
                            // -1 if A can never have zombies
) ;

void GB_macrofy_output
(
    FILE *fp,
    // input:
    const char *cname,      // name of the scalar ... = cij to write
    const char *Cmacro,     // name of the macro is GB_PUT*(Cmacro)
    const char *Cname,      // name of the output matrix
    GrB_Type ctype,         // type of C, ignored if C is iso
    GrB_Type ztype,         // type of cij scalar to cast to ctype write to C
    int csparsity,          // sparsity format of the output matrix
    bool C_iso,             // true if C is iso on output
    bool C_in_iso           // true if C is iso on input
) ;

//------------------------------------------------------------------------------
// monoid identity and terminal values
//------------------------------------------------------------------------------

void GB_enumify_identity       // return enum of identity value
(
    // output:
    int *ecode,             // enumerated identity, 0 to 17 (-1 if fail)
    // input:
    GB_Opcode opcode,       // built-in binary opcode of a monoid
    GB_Type_code zcode      // type code used in the opcode we want
) ;

const char *GB_macrofy_id // return string encoding the value
(
    // input:
    int ecode,          // enumerated identity/terminal value
    size_t zsize,       // size of value
    // output:          // (optional: either may be NULL)
    bool *has_byte,     // true if value is a single repeated byte
    uint8_t *byte       // repeated byte
) ;

void GB_macrofy_bytes
(
    FILE *fp,               // file to write macros, assumed open already
    // input:
    const char *Name,       // all-upper-case name
    const char *variable,   // variable to declaer
    const char *type_name,  // name of the type
    const uint8_t *value,   // array of size nbytes
    size_t nbytes,
    bool is_identity        // true for the identity value
) ;

void GB_enumify_terminal       // return enum of terminal value
(
    // output:
    int *ecode,                 // enumerated terminal, 0 to 31 (-1 if fail)
    // input:
    GB_Opcode opcode,           // built-in binary opcode of a monoid
    GB_Type_code zcode          // type code used in the opcode we want
) ;

//------------------------------------------------------------------------------
// sparsity structure
//------------------------------------------------------------------------------

void GB_enumify_sparsity    // enumerate the sparsity structure of a matrix
(
    // output:
    int *ecode,             // enumerated sparsity structure:
                            // 0:hyper, 1:sparse, 2:bitmap, 3:full
    // input:
    int sparsity            // 0:no matrix, 1:GxB_HYPERSPARSE, 2:GxB_SPARSE,
                            // 4:GxB_BITMAP, 8:GxB_FULL
) ;

void GB_macrofy_sparsity    // construct macros for sparsity structure
(
    // input:
    FILE *fp,
    const char *matrix_name,    // "C", "M", "A", or "B"
    int sparsity
) ;

void GB_macrofy_nvals  
(
    FILE *fp,
    // input:
    const char *Aname,      // name of input matrix (typically A, B, C,..)
    int asparsity,          // sparsity format of the input matrix, -1 if NULL
    bool A_iso              // true if A is iso
) ;

//------------------------------------------------------------------------------
// typedefs, type name and size
//------------------------------------------------------------------------------

void GB_macrofy_typedefs
(
    FILE *fp,
    // input:
    GrB_Type ctype,
    GrB_Type atype,
    GrB_Type btype,
    GrB_Type xtype,
    GrB_Type ytype,
    GrB_Type ztype
) ;

void GB_macrofy_type
(
    FILE *fp,
    // input:
    const char *what,       // typically X, Y, Z, A, B, or C
    const char *what2,      // typically "_" or "2"
    const char *name        // name of the type
) ;

//------------------------------------------------------------------------------
// unary ops
//------------------------------------------------------------------------------

void GB_enumify_apply       // enumerate an apply or tranpose/apply problem
(
    // output:
    uint64_t *scode,        // unique encoding of the entire operation
    // input:
    // C matrix:
    int C_sparsity,         // sparse, hyper, bitmap, or full.  For apply
                            // without transpose, Cx = op(A) is computed where
                            // Cx is just C->x, so the caller uses 'full' when
                            // C is sparse, hyper, or full.
    bool C_is_matrix,       // true for C=op(A), false for Cx=op(A)
    GrB_Type ctype,         // C=((ctype) T) is the final typecast
    // operator:
        const GB_Operator op,       // unary/index-unary to apply; not binaryop
        bool flipij,                // if true, flip i,j for user idxunop
    // A matrix:
    const GrB_Matrix A              // input matrix
) ;

void GB_enumify_unop    // enumify a GrB_UnaryOp or GrB_IndexUnaryOp
(
    // output:
    int *ecode,         // enumerated operator, range 0 to 254
    bool *depends_on_x, // true if the op depends on x
    bool *depends_on_i, // true if the op depends on i
    bool *depends_on_j, // true if the op depends on j
    bool *depends_on_y, // true if the op depends on y
    // input:
    bool flipij,        // if true, then the i and j indices are flipped
    GB_Opcode opcode,   // opcode of GraphBLAS operator to convert into a macro
    GB_Type_code xcode  // op->xtype->code of the operator
) ;

void GB_macrofy_unop
(
    FILE *fp,
    // input:
    const char *macro_name,
    bool flipij,                // if true: op is f(z,x,j,i,y) with ij flipped
    int ecode,
    GB_Operator op              // GrB_UnaryOp or GrB_IndexUnaryOp
) ;

void GB_macrofy_apply           // construct all macros for GrB_apply
(
    // output:
    FILE *fp,                   // target file to write, already open
    // input:
    uint64_t scode,
    // operator:
        const GB_Operator op,       // unary/index-unary to apply; not binaryop
    GrB_Type ctype,
    GrB_Type atype
) ;

uint64_t GB_encodify_apply      // encode an apply problem
(
    // output:
    GB_jit_encoding *encoding,  // unique encoding of the entire problem,
                                // except for the suffix
    char **suffix,              // suffix for user-defined kernel
    // input:
    const GB_jit_kcode kcode,   // kernel to encode
    const int C_sparsity,
    const bool C_is_matrix,     // true for C=op(A), false for Cx=op(A)
    const GrB_Type ctype,
    const GB_Operator op,
    const bool flipij,
    const GrB_Matrix A
) ;

GrB_Info GB_apply_unop_jit      // Cx = op (A), apply unop via the JIT
(
    // output:
    GB_void *Cx,
    // input:
    const GrB_Type ctype,
    const GB_Operator op,       // unary or index unary op
    const bool flipij,          // if true, use z = f(x,j,i,y)
    const GrB_Matrix A,
    const void *ythunk,         // for index unary ops (op->ytype scalar)
    const int64_t *restrict A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB_apply_bind1st_jit   // Cx = op (x,B), apply bind1st via the JIT
(
    // output:
    GB_void *Cx,
    // input:
    const GrB_Type ctype,
    const GrB_BinaryOp binaryop,
    const GB_void *xscalar,
    const GrB_Matrix B,
    const int nthreads
) ;

GrB_Info GB_apply_bind2nd_jit   // Cx = op (x,B), apply bind2nd via the JIT
(
    // output:
    GB_void *Cx,
    // input:
    const GrB_Type ctype,
    const GrB_BinaryOp binaryop,
    const GrB_Matrix A,
    const GB_void *yscalar,
    const int nthreads
) ;

GrB_Info GB_transpose_bind1st_jit
(
    // output:
    GrB_Matrix C,
    // input:
    const GrB_BinaryOp binaryop,
    const GB_void *xscalar,
    const GrB_Matrix A,
    int64_t *restrict *Workspaces,
    const int64_t *restrict A_slice,
    int nworkspaces,
    int nthreads
) ;

GrB_Info GB_transpose_bind2nd_jit
(
    // output:
    GrB_Matrix C,
    // input:
    const GrB_BinaryOp binaryop,
    const GrB_Matrix A,
    const GB_void *yscalar,
    int64_t *restrict *Workspaces,
    const int64_t *restrict A_slice,
    int nworkspaces,
    int nthreads
) ;

GrB_Info GB_transpose_unop_jit  // C = op (A'), transpose unop via the JIT
(
    // output:
    GrB_Matrix C,
    // input:
    GB_Operator op,
    const GrB_Matrix A,
    int64_t *restrict *Workspaces,
    const int64_t *restrict A_slice,
    int nworkspaces,
    int nthreads
) ;

GrB_Info GB_convert_s2b_jit    // convert sparse to bitmap
(
    // output:
    GB_void *Ax_new,
    int8_t *Ab,
    // input:
    GB_Operator op,
    const GrB_Matrix A,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB_concat_sparse_jit      // concatenate A into a sparse matrix C
(
    // input/output
    GrB_Matrix C,
    // input:
    int64_t cistart,
    const GB_Operator op,
    const GrB_Matrix A,
    int64_t *restrict W,
    const int64_t *restrict A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB_concat_full_jit      // concatenate A into a full matrix C
(
    // input/output
    GrB_Matrix C,
    // input:
    int64_t cistart,
    int64_t cvstart,
    const GB_Operator op,
    const GrB_Matrix A,
    const int A_nthreads
) ;

GrB_Info GB_concat_bitmap_jit      // concatenate A into a bitmap matrix C
(
    // input/output
    GrB_Matrix C,
    // input:
    int64_t cistart,
    int64_t cvstart,
    const GB_Operator op,
    const GrB_Matrix A,
    GB_Werk Werk
) ;

GrB_Info GB_split_sparse_jit      // split A into a sparse tile C
(
    // input/output
    GrB_Matrix C,
    // input:
    const GB_Operator op,
    const GrB_Matrix A,
    int64_t akstart,
    int64_t aistart,
    int64_t *restrict Wp,
    const int64_t *restrict C_ek_slicing,
    const int C_ntasks,
    const int C_nthreads
) ;

GrB_Info GB_split_full_jit      // split A into a full tile C
(
    // input/output
    GrB_Matrix C,
    // input:
    const GB_Operator op,
    const GrB_Matrix A,
    int64_t avstart,
    int64_t aistart,
    const int C_nthreads
) ;

GrB_Info GB_split_bitmap_jit      // split A into a bitmap tile C
(
    // input/output
    GrB_Matrix C,
    // input:
    const GB_Operator op,
    const GrB_Matrix A,
    int64_t avstart,
    int64_t aistart,
    const int C_nthreads
) ;

//------------------------------------------------------------------------------
// builder kernel
//------------------------------------------------------------------------------

uint64_t GB_encodify_build      // encode an build problem
(
    // output:
    GB_jit_encoding *encoding,  // unique encoding of the entire problem,
                                // except for the suffix
    char **suffix,              // suffix for user-defined kernel
    // input:
    const GB_jit_kcode kcode,   // kernel to encode
    const GrB_BinaryOp dup,     // operator for summing up duplicates
    const GrB_Type ttype,       // type of Tx array
    const GrB_Type stype        // type of Sx array
) ;

void GB_enumify_build       // enumerate a GB_build problem
(
    // output:
    uint64_t *build_code,   // unique encoding of the entire operation
    // input:
    GrB_BinaryOp dup,       // operator for duplicates
    GrB_Type ttype,         // type of Tx
    GrB_Type stype          // type of Sx
) ;

void GB_macrofy_build           // construct all macros for GB_build
(
    // output:
    FILE *fp,                   // target file to write, already open
    // input:
    uint64_t build_code,        // unique encoding of the entire problem
    GrB_BinaryOp dup,           // dup binary operator to macrofy
    GrB_Type ttype,             // type of Tx
    GrB_Type stype              // type of Sx
) ;

GrB_Info GB_build_jit               // GB_builder JIT kernel
(
    // output:
    GB_void *restrict Tx,
    int64_t *restrict Ti,
    // input:
    const GB_void *restrict Sx,
    const GrB_Type ttype,           // type of Tx
    const GrB_Type stype,           // type of Sx
    const GrB_BinaryOp dup,         // operator for summing duplicates
    const int64_t nvals,            // number of tuples
    const int64_t ndupl,            // number of duplicates
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
) ;

//------------------------------------------------------------------------------
// select kernel
//------------------------------------------------------------------------------

uint64_t GB_encodify_select     // encode an select problem
(
    // output:
    GB_jit_encoding *encoding,  // unique encoding of the entire problem,
                                // except for the suffix
    char **suffix,              // suffix for user-defined kernel
    // input:
    const GB_jit_kcode kcode,   // kernel to encode
    const bool C_iso,
    const bool in_place_A,
    const GrB_IndexUnaryOp op,
    const bool flipij,
    const GrB_Matrix A
) ;

void GB_enumify_select      // enumerate a GrB_selectproblem
(
    // output:
    uint64_t *scode,        // unique encoding of the entire operation
    // input:
    bool C_iso,
    bool in_place_A,
    // operator:
    GrB_IndexUnaryOp op,    // the index unary operator to enumify
    bool flipij,            // if true, flip i and j
    // A matrix:
    GrB_Matrix A
) ;

void GB_macrofy_select          // construct all macros for GrB_select
(
    // output:
    FILE *fp,                   // target file to write, already open
    // input:
    uint64_t scode,
    // operator:
    const GrB_IndexUnaryOp op,
    GrB_Type atype
) ;

GrB_Info GB_select_bitmap_jit      // select bitmap
(
    // output:
    int8_t *Cb,
    int64_t *cnvals_handle,
    // input:
    const bool C_iso,
    const GrB_Matrix A,
    const bool flipij,
    const GB_void *restrict ythunk,
    const GrB_IndexUnaryOp op,
    const int nthreads
) ;

GrB_Info GB_select_phase1_jit      // select phase1
(
    // output:
    int64_t *restrict Cp,
    int64_t *restrict Wfirst,
    int64_t *restrict Wlast,
    // input:
    const bool C_iso,
    const bool in_place_A,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const GrB_IndexUnaryOp op,
    const bool flipij,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB_select_phase2_jit      // select phase2
(
    // output:
    int64_t *restrict Ci,
    GB_void *restrict Cx,                   // NULL if C is iso-valued
    // input:
    const int64_t *restrict Cp,
    const bool C_iso,
    const bool in_place_A,
    const int64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const bool flipij,
    const GB_void *restrict ythunk,
    const GrB_IndexUnaryOp op,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

//------------------------------------------------------------------------------
// assign/subassign kernel
//------------------------------------------------------------------------------

void GB_enumify_assign      // enumerate a GrB_assign problem
(
    // output:
    uint64_t *scode,        // unique encoding of the entire operation
    // input:
    // C matrix:
    GrB_Matrix C,
    bool C_replace,
    // index types:
    int Ikind,              // 0: all (no I), 1: range, 2: stride, 3: list
    int Jkind,              // ditto
    // M matrix:
    GrB_Matrix M,           // may be NULL
    bool Mask_struct,       // mask is structural
    bool Mask_comp,         // mask is complemented
    // operator:
    GrB_BinaryOp accum,     // the accum operator (may be NULL)
    // A matrix
    GrB_Matrix A,           // NULL for scalar assignment
    GrB_Type scalar_type,
    int assign_kind         // 0: assign, 1: subassign, 2: row, 3: col
) ;

void GB_macrofy_assign          // construct all macros for GrB_assign
(
    // output:
    FILE *fp,                   // target file to write, already open
    // input:
    uint64_t scode,
    GrB_BinaryOp accum,         // accum operator to macrofy
    GrB_Type ctype,
    GrB_Type atype              // matrix or scalar type
) ;

uint64_t GB_encodify_assign     // encode an assign problem
(
    // output:
    GB_jit_encoding *encoding,  // unique encoding of the entire problem,
                                // except for the suffix
    char **suffix,              // suffix for user-defined kernel
    // input:
    const GB_jit_kcode kcode,   // kernel to encode
    // C matrix:
    GrB_Matrix C,
    bool C_replace,
    // index types:
    int Ikind,              // 0: all (no I), 1: range, 2: stride, 3: list
    int Jkind,              // ditto
    // M matrix:
    GrB_Matrix M,           // may be NULL
    bool Mask_struct,       // mask is structural
    bool Mask_comp,         // mask is complemented
    // operator:
    GrB_BinaryOp accum,     // the accum operator (may be NULL)
    // A matrix or scalar
    GrB_Matrix A,           // NULL for scalar assignment
    GrB_Type scalar_type,
    int assign_kind         // 0: assign, 1: subassign, 2: row, 3: col
) ;

GrB_Info GB_subassign_jit
(
    // input/output:
    GrB_Matrix C,
    // input:
    const bool C_replace,
    // I:
    const GrB_Index *I,
    const int64_t ni,
    const int64_t nI,
    const int Ikind,
    const int64_t Icolon [3],
    // J:
    const GrB_Index *J,
    const int64_t nj,
    const int64_t nJ,
    const int Jkind,
    const int64_t Jcolon [3],
    // mask M:
    const GrB_Matrix M,
    const bool Mask_comp,
    const bool Mask_struct,
    // accum, if present:
    const GrB_BinaryOp accum,   // may be NULL
    // A matrix or scalar:
    const GrB_Matrix A,         // NULL for scalar assignment
    const void *scalar,
    const GrB_Type scalar_type,
    // kind and kernel:
    const int assign_kind,      // row assign, col assign, assign, or subassign
    const int assign_kernel,    // GB_JIT_KERNEL_SUBASSIGN_01, ... etc
    const char *kname,          // kernel name
    GB_Werk Werk
) ;

//------------------------------------------------------------------------------
// macrofy a user operator or type as its own kernel
//------------------------------------------------------------------------------

void GB_macrofy_user_op         // construct a user-defined operator
(
    // output:
    FILE *fp,                   // target file to write, already open
    // input:
    const GB_Operator op        // op to construct in a JIT kernel
) ;

uint64_t GB_encodify_user_op      // encode a user defined op
(
    // output:
    GB_jit_encoding *encoding,  // unique encoding of the entire problem,
                                // except for the suffix
    char **suffix,              // suffix for user-defined kernel
    // input:
    const GB_Operator op
) ;

GrB_Info GB_user_op_jit         // construct a user operator in a JIT kernel
(
    // output:
    void **user_function,       // function pointer
    // input:
    const GB_Operator op        // unary, index unary, or binary op
) ;

void GB_macrofy_user_type       // construct a user-defined type
(
    // output:
    FILE *fp,                   // target file to write, already open
    // input:
    const GrB_Type type         // type to construct in a JIT kernel
) ;

uint64_t GB_encodify_user_type      // encode a user defined type
(
    // output:
    GB_jit_encoding *encoding,  // unique encoding of the entire problem,
                                // except for the suffix
    char **suffix,              // suffix for user-defined kernel
    // input:
    const GrB_Type type
) ;

GrB_Info GB_user_type_jit       // construct a user type in a JIT kernel
(
    // output:
    size_t *typesize,           // sizeof the type
    // input:
    const GrB_Type type         // user-defined type
) ;

//------------------------------------------------------------------------------
// macrofy for all methods
//------------------------------------------------------------------------------

void GB_macrofy_family
(
    // output:
    FILE *fp,                   // target file to write, already open
    // input:
    GB_jit_family family,       // family to macrofy
    uint64_t scode,             // encoding of the specific problem
    GrB_Semiring semiring,      // semiring (for mxm family only)
    GrB_Monoid monoid,          // monoid (for reduce family only)
    GB_Operator op,             // unary/index_unary/binary op
    GrB_Type type1,
    GrB_Type type2,
    GrB_Type type3
) ;

#endif

