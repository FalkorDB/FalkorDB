function test10_compare (op, C1, C2, tol)
%TEST10_COMPARE check results for test10
%
% Compare results for test10, results from unary operators.
% Usage: test10_compare (op, C1, C2, tol) ;
%
% acos, asin, and other a* trig functions can return different but valid
% results.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C1 = GB_spec_matrix (C1, 0) ;
C2 = GB_spec_matrix (C2, 0) ;
assert (isequal (C1.pattern, C2.pattern)) ;

[i,j,x] = find (C2.matrix) ;
[m,n] = size (C2.matrix) ;
x = ones (size (x)) ;
S = sparse (i,j,x,m,n) ;

id = test_cast (0, C1.class) ;

C1.matrix = double (C1.matrix) ;
if (isequal (C1.class, 'single complex'))
    C1.class = 'double complex' ;
else
    C1.class = 'double' ;
end

C2.matrix = double (C2.matrix) ;
if (isequal (C2.class, 'single complex'))
    C2.class = 'double complex' ;
else
    C2.class = 'double' ;
end

switch (op.opname)

    case { 'acos' }
        C1.matrix = cos (C1.matrix) .* S ;
        C2.matrix = cos (C2.matrix) .* S ;
        GB_spec_compare (C1, C2, id, tol) ;

    case { 'asin' }
        C1.matrix = sin (C1.matrix) .* S ;
        C2.matrix = sin (C2.matrix) .* S ;
        GB_spec_compare (C1, C2, id, tol) ;

    case { 'atan' }
        C1.matrix = tan (C1.matrix) .* S ;
        C2.matrix = tan (C2.matrix) .* S ;
        GB_spec_compare (C1, C2, id, tol) ;

    case { 'acosh' }
        C1.matrix = cosh (C1.matrix) .* S ;
        C2.matrix = cosh (C2.matrix) .* S ;
        GB_spec_compare (C1, C2, id, tol) ;

    case { 'asinh' }
        C1.matrix = sinh (C1.matrix) .* S ;
        C2.matrix = sinh (C2.matrix) .* S ;
        GB_spec_compare (C1, C2, id, tol) ;

    case { 'atanh' }
        C1.matrix = tanh (C1.matrix) .* S ;
        C2.matrix = tanh (C2.matrix) .* S ;
        GB_spec_compare (C1, C2, id, tol) ;

    otherwise
        GB_spec_compare (C1, C2, id, tol) ;
end

