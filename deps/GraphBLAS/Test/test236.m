function test236
%TEST236 test GxB_Matrix_sort and GxB_Vector_sort

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
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

            P1 = GB_mex_Matrix_sort  (lt, A, [ ], 1, 'int32') ;
            assert (isequal (P1.class, 'int32')) ;
            P1.class = 'int64' ;
            GB_spec_compare (P1, P2) ;

            P1 = GB_mex_Matrix_sort  (lt, A, [ ], 1, 'uint32') ;
            assert (isequal (P1.class, 'uint32')) ;
            P1.class = 'int64' ;
            GB_spec_compare (P1, P2) ;

            try
                P1 = GB_mex_Matrix_sort  (lt, A, [ ], 1, 'double') ;
                ok = 0 ;
            catch me
                ok = 1 ;
            end
            assert (ok) ;

            P1 = GB_mex_Matrix_sort  (lt, A, [ ], 1, 'uint64') ;
            assert (isequal (P1.class, 'uint64')) ;
            P1.class = 'int64' ;
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

            [C1,P1] = GB_mex_Matrix_sort  (lt, A, desc, -1) ;
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
is_csc = 1 ;

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

% with typecasing, to bool 
lt.optype = 'bool' ;

    A.matrix = double (A.matrix > 0) ;
    A.pattern = logical (spones (A.matrix)) ;

    fprintf ('.') ;
    [C1,P1] = GB_mex_Matrix_sort  (lt, A) ;
    [C2,P2] = GB_spec_Matrix_sort (lt, A, [ ]) ;
    GB_spec_compare (C1, C2) ;
    GB_spec_compare (P1, P2) ;

lt.opname = 'lt' ;
gt.opname = 'gt' ;
lt.optype = 'double' ;
gt.optype = 'double' ;

% matrix with large vectors
fprintf (' large') ;
m = 100000 ;
n = 2 ;
A = sparse (rand (m, n)) ;
A (:,2) = sprand (m, 1, 0.02) ;

fprintf ('.') ;
[C1,P1] = GB_mex_Matrix_sort  (lt, A, desc) ;
[C2,P2] = GB_spec_Matrix_sort (lt, A, desc) ;
GB_spec_compare (C1, C2) ;
GB_spec_compare (P1, P2) ;

fprintf ('\ntest236: all tests passed\n') ;

