function test16
%TEST16 test user-defined complex type (runs all testc*.m)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% all complex matrix tests

for k = [false true]
    GB_builtin_complex_set (k) ;
    if (k)
        fprintf ('\nTesting built-in GxB_FC64 complex type:\n') ;
    else
        fprintf ('\nTesting user-defined Complex type:\n') ;
    end

    testc1 (k)      % test ops
    testc2 (0,k)    % A'*B, A+B, A*B
    testc3 (k)      % extract column, extract submatrix
    testc4 (k)      % extractElement, setElement
    testc5 (k)      % subref
    testc6 (k)      % apply
    testc7 (k)      % assign
    testc8 (k)      % eWise
    testc9 (k)      % extractTuples
    testca (k)      % mxm, mxv, vxm
    testcb (k)      % reduce
    testcc (k)      % transpose
end

GB_builtin_complex_set (true) ;
fprintf ('\ntest16: all complex tests passed\n') ;

