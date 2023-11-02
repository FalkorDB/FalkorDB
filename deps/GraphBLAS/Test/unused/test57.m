function test57 (op)
%TEST57 test operator on large uint32 values
%
% Usage:
%   test57(op)
%   test57      % Default op is 'max' if no arguments given

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin < 1)
    op = 'max'
end

GB_mex_op (op, uint32(4294967211), uint32(4294967203))

fprintf ('\ntest57: all tests passed\n') ;

