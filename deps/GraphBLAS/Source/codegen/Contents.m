% SuiteSparse/GraphBLAS/Source
%
% These files are used to create the files in GraphBLAS/Source/mxm/*__* and
% GraphBLAS/FactoryKernels, from the input files in Source/codegen/Generator.
% These functions do not need to be used by the end user.
%
%   codegen                      - generate all code for ../../FactoryKernels/*.c
%   codegen_aop                  - create functions for all binary operators for assign/subassign
%   codegen_aop_method           - create a function to compute C(:,:)+=A
%   codegen_aop_template         - create aop functions
%   codegen_as                   - create functions for assign/subassign methods with no accum
%   codegen_as_template          - create a function for subassign/assign with no accum
%   codegen_axb                  - create all C=A*B functions for all semirings
%   codegen_axb_compare_template - create a function for a semiring with a TxT -> bool multiplier
%   codegen_axb_method           - create a function to compute C=A*B over a semiring
%   codegen_axb_template         - create a function for a semiring with a TxT->T multiplier
%   codegen_contains             - same as contains (text, pattern)
%   codegen_ew                   - create ewise kernels
%   codegen_ew_method            - create an ewise kernel
%   codegen_ew_template          - create ewise kernels
%   codegen_red                  - create functions for all reduction operators
%   codegen_red_method           - create a reduction function, C = reduce (A)
%   codegen_sel                  - create functions for all selection operators
%   codegen_sel_method           - create a selection function, C = select (A,thunk)
%   codegen_type                 - determine function fname, signed or not
%   codegen_uop                  - create functions for all unary operators
%   codegen_uop_identity         - create identity functions
%   codegen_uop_method           - create a function to compute C=uop(A)
%   codegen_uop_template         - CODEGEN_UNOP_TEMPLATE create uop functions

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

