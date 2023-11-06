function test209
%TEST209 test iso build

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;
GB_builtin_complex_set (true) ;

[~, ~, ~, types ~, ~,] = GB_spec_opsall ;
types = types.all ;
op.opname = 'second' ;

% GB_mex_burble (1) ;

n = 100 ;
m = 200 ;
nnz = 1000 ;

I = irand (0, m-1, nnz, 1) ;
I1 = double (I) + 1 ;
J = irand (0, n-1, nnz, 1) ;
Y = 10 * rand (nnz, 1) ;
Yimag = 10 * rand (nnz, 1) ;

ntypes = 1+length(types) ;
for k = 1:ntypes

    type = types {min (k, length (types))} ;

    fprintf ('%s ', type) ;
    if (test_contains (type, 'complex') && k <= length (types)) ;
        % the build is generic if Y is double and op is complex
        Y = Y + 1i * Yimag ;
    end
    X = GB_mex_cast (Y, type) ;

    Z = X (1) ;
    X (:) = Z ;

    op.optype = type ;

    % non-iso matrix build
    C1 = GB_mex_Matrix_build (I, J, X, m, n, op, type) ;
    C2 = GB_spec_build (I, J, X, m, n, op) ;
    GB_spec_compare (C1, C2) ;

    % iso matrix build
    C1 = GB_mex_Matrix_build (I, J, Z, m, n, op, type) ;
    C2 = GB_spec_build (I, J, X, m, n, op) ;
    GB_spec_compare (C1, C2) ;

    % non-iso vector build
    C1 = GB_mex_Vector_build (I, X, m, op, type) ;
    C2 = GB_spec_build (I, [ ], X, m, 1, op) ;
    GB_spec_compare (C1, C2) ;

    % iso vector build
    C1 = GB_mex_Vector_build (I, Z, m, op, type) ;
    C2 = GB_spec_build (I, [ ], X, m, 1, op) ;
    GB_spec_compare (C1, C2) ;

end

% large iso vector build
n = 1 ;
m = 200e6 ;
nnz = 4e6 ;
X = true (nnz, 1) ;
Z = X (1) ;

I = irand (0, m-1, nnz, 1) ;
I1 = double (I) + 1 ;
C1 = GB_mex_Vector_build (I, Z, m, op, type) ;
C2 = logical (sparse (I1, 1, X, m, 1)) ;
assert (isequal (C1.matrix, C2)) ;

GB_mex_burble (0) ;
fprintf ('\ntest209: all tests passed\n') ;

