function test210
%TEST209 test iso build

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;
GB_builtin_complex_set (true) ;

[~, ~, ~, types ~, ~,] = GB_spec_opsall ;
types = types.all ;
op.opname = 'second' ;

n = 100 ;
m = 200 ;
nnz = 1000 ;

I = irand (0, m-1, nnz, 1) ;
I1 = double (I) + 1 ;
J = irand (0, n-1, nnz, 1) ;
Y = 10 * rand (nnz, 1) ;
Yimag = 10 * rand (nnz, 1) ;

% desc.*list can be: use_values, use_indices, or is_stride,
% but is_stride cannot be used by build
desc.rowindex_list = 'use_values' ;
desc.colindex_list = 'use_values' ;
desc.value_list = 'use_values' ;

desc1 = desc ;
desc1.value_list = 'use_indices' ;

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
    R = (0:length (X)-1)' ;

    op.optype = type ;

    % non-iso matrix build
    C1 = GB_mex_Matrix_build (I, J, X, m, n, op, type) ;
    C2 = GB_spec_build (I, J, X, m, n, op) ;
    GB_spec_compare (C1, C2) ;

    % non-iso matrix build, using _Vector method
    C1 = GB_mex_Matrix_build (I, J, X, m, n, op, type, 1, desc) ;
    GB_spec_compare (C1, C2) ;

    % iso matrix build
    C1 = GB_mex_Matrix_build (I, J, Z, m, n, op, type) ;
    C2 = GB_spec_build (I, J, X, m, n, op) ;
    GB_spec_compare (C1, C2) ;

    % iso matrix build, using _Vector method
    C1 = GB_mex_Matrix_build (I, J, Z, m, n, op, type, 1, desc) ;
    GB_spec_compare (C1, C2) ;

    % iso matrix build, using _Vector method
    clear Xvector
    Xvector.matrix = X ;
    Xvector.iso = true ;
    for sparsity = [2 4 8]
        Xvector.sparsity = sparsity ;
        C1 = GB_mex_Matrix_build (I, J, Xvector, m, n, op, type, 1, desc) ;
        GB_spec_compare (C1, C2) ;
    end

    % non-iso vector build
    C1 = GB_mex_Vector_build (I, X, m, op, type) ;
    C2 = GB_spec_build (I, [ ], X, m, 1, op) ;
    GB_spec_compare (C1, C2) ;

    % non-iso vector build, using _Vector method
    C1 = GB_mex_Vector_build (I, X, m, op, type, desc) ;
    GB_spec_compare (C1, C2) ;

    % iso vector build
    C1 = GB_mex_Vector_build (I, Z, m, op, type) ;
    C2 = GB_spec_build (I, [ ], X, m, 1, op) ;
    GB_spec_compare (C1, C2) ;

    % iso vector build, using _Vector method
    C1 = GB_mex_Vector_build (I, Z, m, op, type, desc) ;
    GB_spec_compare (C1, C2) ;

    if (isequal (type, 'double') || isequal (type, 'uint32'))
        % non-iso matrix build, ramp
        C1 = GB_mex_Matrix_build (I, J, R, m, n, op, type) ;
        C2 = GB_spec_build (I, J, R, m, n, op) ;
        GB_spec_compare (C1, C2) ;

        % non-iso matrix build, ramp, using _Vector method, and indices of X
        C3 = GB_mex_Matrix_build (I, J, X, m, n, op, type, 1, desc1) ;
        GB_spec_compare (C3, C2) ;

        % non-iso matrix build, ramp, using _Vector method, and indices of Xvector
        clear Xvector
        Xvector.matrix = X ;
        for sparsity = [2 4 8]
            Xvector.sparsity = sparsity ;
            C3 = GB_mex_Matrix_build (I, J, Xvector, m, n, op, type, 1, desc1) ;
        end
        GB_spec_compare (C3, C2) ;
    end
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

fprintf ('\ntest209: all tests passed\n') ;

