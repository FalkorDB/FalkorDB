function gbtest104
%GBTEST104 test formats

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;
A = GrB (rand (4), 'sparse') %#ok<*NOPRT>
A = GrB (A, 'hypersparse')
A = GrB (A, 'bitmap')
A = GrB (A, 'full') %#ok<*NASGU>

fprintf ('\ngbtest104: all tests passed\n') ;

