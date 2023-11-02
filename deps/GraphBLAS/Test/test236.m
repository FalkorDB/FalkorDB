function test236
%TEST236 test GxB_Matrix_sort and GxB_Vector_sort

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[~, ~, ~, types, ~, ~] = GB_spec_opsall ;
types = types.all ;

fprintf ('test236 -----------GxB_Matrix_sort and GxB_Vector_sort\n') ;

m = 20 ;
n = 10 ;

rng ('default') ;

lt.opname = 'lt' ;
lt.optype = 'none' ;

gt.opname = 'gt' ;
gt.optype = 'none' ;

desc.inp0 = 'tran' ;

for k = 1:length (types)
    type = types {k} ;
    if (test_contains (type, 'complex'))
        continue
    end
    fprintf (' %s', type) ;
    lt.optype = type ;
    gt.optype = type ;

    for is_csc = 0:1
      for density = [0.3 inf]

        A = GB_spec_random (m, n, density, 100, type, is_csc) ;

        for c = [1 2 4 8]
            A.sparsity = c ;
            fprintf ('.') ;

            C1 = GB_mex_Matrix_sort  (lt, A) ;
            C2 = GB_spec_Matrix_sort (lt, A, [ ]) ;
            GB_spec_compare (C1, C2) ;

            P1 = GB_mex_Matrix_sort  (lt, A, [ ], 1) ;
            [C2,P2] = GB_spec_Matrix_sort (lt, A, [ ]) ;
            GB_spec_compare (P1, P2) ;

            C1 = GB_mex_Matrix_sort  (gt, A) ;
            C2 = GB_spec_Matrix_sort (gt, A, [ ]) ;
            GB_spec_compare (C1, C2) ;

            P1 = GB_mex_Matrix_sort  (gt, A, [ ], 1) ;
            [C2,P2] = GB_spec_Matrix_sort (gt, A, [ ]) ;
            GB_spec_compare (P1, P2) ;

            [C1,P1] = GB_mex_Matrix_sort  (gt, A) ;
            [C2,P2] = GB_spec_Matrix_sort (gt, A, [ ]) ;
            GB_spec_compare (C1, C2) ;
            GB_spec_compare (P1, P2) ;

            C1 = GB_mex_Matrix_sort  (lt, A, desc) ;
            C2 = GB_spec_Matrix_sort (lt, A, desc) ;
            GB_spec_compare (C1, C2) ;

            [C1,P1] = GB_mex_Matrix_sort  (lt, A, desc) ;
            [C2,P2] = GB_spec_Matrix_sort (lt, A, desc) ;
            GB_spec_compare (C1, C2) ;
            GB_spec_compare (P1, P2) ;

            [C1,P1] = GB_mex_Matrix_sort  (gt, A, desc) ;
            [C2,P2] = GB_spec_Matrix_sort (gt, A, desc) ;
            GB_spec_compare (C1, C2) ;
            GB_spec_compare (P1, P2) ;

            C1 = GB_mex_Matrix_sort  (gt, A, desc) ;
            C2 = GB_spec_Matrix_sort (gt, A, desc) ;
            GB_spec_compare (C1, C2) ;

        end
      end
    end

    for density = [0.3 inf]
        A = GB_spec_random (m, 1, density, 100, type, true) ;
        fprintf ('.') ;

        for c = [1 2 4 8]
            A.sparsity = c ;
            fprintf ('.') ;

            [C1,P1] = GB_mex_Vector_sort  (lt, A) ;
            [C2,P2] = GB_spec_Vector_sort (lt, A, [ ]) ;
            GB_spec_compare (C1, C2) ;
            GB_spec_compare (P1, P2) ;

            [C1,P1] = GB_mex_Vector_sort  (gt, A) ;
            [C2,P2] = GB_spec_Vector_sort (gt, A, [ ]) ;
            GB_spec_compare (C1, C2) ;
            GB_spec_compare (P1, P2) ;

            [C1,P1] = GB_mex_Vector_sort  (lt, A, desc) ;
            [C2,P2] = GB_spec_Vector_sort (lt, A, desc) ;
            GB_spec_compare (C1, C2) ;
            GB_spec_compare (P1, P2) ;

            [C1,P1] = GB_mex_Vector_sort  (gt, A, desc) ;
            [C2,P2] = GB_spec_Vector_sort (gt, A, desc) ;
            GB_spec_compare (C1, C2) ;
            GB_spec_compare (P1, P2) ;
        end
    end
    
end

% iso cases
fprintf (' iso') ;
lt.optype = 'double' ;
clear A
A.matrix = pi * spones (sprand (m, n, 0.4)) ;
A.iso = true ;

fprintf ('.') ;
C1 = GB_mex_Matrix_sort  (lt, A) ;
C2 = GB_spec_Matrix_sort (lt, A, [ ]) ;
GB_spec_compare (C1, C2) ;

fprintf ('.') ;
[C1,P1] = GB_mex_Matrix_sort  (lt, A, desc) ;
[C2,P2] = GB_spec_Matrix_sort (lt, A, desc) ;
GB_spec_compare (C1, C2) ;
GB_spec_compare (P1, P2) ;

% matrix with one entry
fprintf (' one_entry.') ;
clear A
A = sparse (m, n) ;
A (2,3) = 42 ;
C1 = GB_mex_Matrix_sort  (lt, A) ;
C2 = GB_spec_Matrix_sort (lt, A, [ ]) ;
GB_spec_compare (C1, C2) ;

% with typecasting
fprintf (' typecast') ;
lt.optype = 'single' ;
gt.optype = 'single' ;
A = GB_spec_random (m, n, 0.3, 100, 'double', is_csc) ;

fprintf ('.') ;
[C1,P1] = GB_mex_Matrix_sort  (lt, A) ;
[C2,P2] = GB_spec_Matrix_sort (lt, A, [ ]) ;
GB_spec_compare (C1, C2) ;
GB_spec_compare (P1, P2) ;

fprintf ('.') ;
[C1,P1] = GB_mex_Matrix_sort  (gt, A) ;
[C2,P2] = GB_spec_Matrix_sort (gt, A, [ ]) ;
GB_spec_compare (C1, C2) ;
GB_spec_compare (P1, P2) ;

% matrix with large vectors
fprintf (' large') ;
m = 100000 ;
n = 2 ;
A = sparse (rand (m, n)) ;
A (:,2) = sprand (m, 1, 0.02) ;
lt.optype = 'double' ;

fprintf ('.') ;
[C1,P1] = GB_mex_Matrix_sort  (lt, A, desc) ;
[C2,P2] = GB_spec_Matrix_sort (lt, A, desc) ;
GB_spec_compare (C1, C2) ;
GB_spec_compare (P1, P2) ;

fprintf ('\ntest236: all tests passed\n') ;

