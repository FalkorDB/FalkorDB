function test33
%TEST33 test a semiring

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

semiring = struct ( ...
    'multiply', 'times', ...
    'add', 'plus', ...
    'class', 'double' )

GB_mex_semiring (semiring)

fprintf ('\ntest33: all tests passed\n') ;

