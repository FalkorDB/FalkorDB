SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
SPDX-License-Identifier: Apache-2.0

This folder, GraphBLAS/Source, contains all the primary source files for
GraphBLAS, and internal include files that are meant for internal inclusion in
GraphBLAS itself.  They should not be included in end-user applications.

The Source/*/template/* files are not compiled separately, but are #include'd
into files in this folder instead.  The are templatized source files.  The
Source/*/include/* files are header files for template methods.  Both template
and include files are merged into the JITpackage and then exported to
~/.SuiteSparse/GrB.x.y.z/src.

Source/*/factory/* files are not used in the JIT.

Files and folders in Source/

add                 GrB_eWiseAdd and GxB_eWiseUnion (see also ewise folder)

aliased             methods for determining if any components of a matrix
                    are aliased (shared) with another matrix.  Such a matrix
                    is called shallow (an array such as A->i is shallow
                    if A->i_shallow is true, and is owned by another matrix).
                    Such matrices are never returned to the user; they are
                    only internal.

apply               GrB_apply

assign              GrB_assign and GxB_assign

binaryop            GrB_BinaryOp object

builder             GrB_build

builtin             builtin types, operators, monoids, and semirings

callback            JIT kernels are not linked against -lgraphblas and thus
                    cannot call back into GraphBLAS directly.  Instead, a
                    struct containing function pointers to various utility
                    functions is passed to each JIT kernel.

cast                typecasting

clear               GrB_clear

codegen             MATLAB scripts for creating ../FactoryKernels,
                    Source/mxm/GB_AxB__any_pair_iso.c, and
                    Source/mxm/GB_AxB__include1.h.

compatible          testing if operators and types are compatible (if they can
                    be typecasted to another type.

concat              GxB_concat

context             GxB_Context object

convert             convert between matrix formats (sparse, hypersparse, bitmap,
                    and full), and conform a matrix to its desired format.

cpu                 wrapper for Google's cpu_features package

cumsum              cumulative sum

descriptor          GrB_Descriptor object

diag                GrB_diag and GxB_diag

dup                 GrB_dup

element             GrB_setElement, GrB_extractElement, GrB_removeElement

emult               GrB_eWiseMult (see also ewise folder)

ewise               element-wise methods for ewise add, emult, and union

extract             GrB_extract

extractTuples       GrB_extractTuples

gateway             definitions for calling methods in the CUDA folder

GB.h                primary internal include file

GB_control.h        controls which FactoryKernels are compiled

generic             definitions for generic methods

get_set             GrB_get, GrB_set

global              global variables and the routines for accessing them

helper              methods used in MATLAB/Octave @GrB interface only

hyper               methods for hypersparse matrices

ij                  determining properities of I and J index lists for
                    GrB_extract, GrB_assign, and GxB_subassign

import_export       GrB_import, GrB_export, GxB_import, and GxB_export

include             general-purpose header files that do not fit into any
                    particular subfolder of GraphBLAS/Source, such as compiler
                    settings, and GB_include.h which is a primary internal
                    include file.

indexbinaryop       GxB_IndexBinaryOp object

indexunaryop        GrB_IndexUnaryOp object

init                GrB_init, GxB_init, GrB_finalize, GrB_error, GrB_getVersion

iso                 iso-valued matrices

iterator            GxB_Iterator object and its methods

jitifyer            the GraphBLAS Just-In-Time compiler for CPU and CUDA JIT
                    kernels

jit_kernels         templates for all CPU JIT kernels.  These are not compiled
                    when the libgraphblas.so library is built.  Instead, they
                    are compiled at run time when requested by the JIT.

jit_wrappers        interfaces that access the JIT to call/load/compile each
                    CPU JIT kernel

kronecker           GrB_kronecker

lz4_wrapper         wrapper for the lz4 compression package

mask                mask/accum methods, for computing C<M>+=T

math                basic mathematical functions

matrix              GrB_Matrix object

memory              memory allocators (wrappers for malloc, free, etc),
                    memset, memcpy

monoid              GrB_Monoid object

mxm                 GrB_mxm, GrB_vxm, and GrB_mxv

nvals               GrB_nvals

ok                  debugging assertions

omp                 OpenMP interface and atomics

op                  methods for operators (GrB_UnaryOp, GrB_BinaryOp, etc)

pack_unpack         GxB_pack, GxB_unpack

pending             pending tuples for updates to matrices, vectors, and scalars

positional          methods for positional operators

print               GxB_print, GxB_fprint

README.txt          this file

reduce              GrB_reduce, to scalar and vector

reshape             GrB_reshape

resize              GrB_resize

scalar              GrB_Scalar object

select              GrB_select

semiring            GrB_Semiring object

serialize           GrB_serialize, GrB_deserialze

slice               methods for slicing matrices to create parallel tasks

sort                GxB_sort, and internal sorting methods

split               GxB_split

transplant          transplant contents of one matrix into another

transpose           GrB_transpose

type                GrB_Type object

unaryop             GrB_UnaryOp object

vector              GrB_Vector object

wait                GrB_wait

werk                the Werk space is a small amount of space on the stack
                    use for small things (scalars, O(# parallel OpenMP tasks),
                    and such.  It is spell differently for easier 'grep'.

zstd_wrapper        wrapper for the zstd compression package


