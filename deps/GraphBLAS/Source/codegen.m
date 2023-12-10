function codegen
%CODEGEN generate all code for FactoryKernels/*.c
%
% This code generation method works on octave7 and MATLAB.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

codegen_as ;        % subassign/assign with no accum
codegen_axb ;       % semirings
codegen_ew ;        % ewise kernels: binary op and accum
codegen_aop ;       % subassign/assign kernels with accum
codegen_unop ;      % unary operators
codegen_red ;       % monoids
codegen_sel ;       % select operators


