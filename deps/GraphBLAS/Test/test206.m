function test206
%TEST206 test iso select

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[~, ~, ~, types, ~, select_ops] = GB_spec_opsall ;
types = types.all ;

% GB_mex_burble (1) ;
rng ('default') ;
n = 10 ;
Cin = sparse (n,n) ;

A.matrix = spones (sprand (n, n, 0.5)) ;
A.iso = true ;

% sparse/hypersparse/bitmap case
for k2 = 1:length (select_ops)
    op = select_ops {k2} ;
    fprintf ('%s ', op) ;
    for s = [1 2 4]
        A.sparsity = s ;
        for k1 = 1:length (types)
            type = types {k1} ;
            A.class = type ;
            if (test_contains (type, 'complex'))
                continue ;
            end
            for Thunk = -1:1
                k = sparse (Thunk) ;
                C1 = GB_mex_select  (Cin, [ ], [ ], op, A, k, [ ]) ;
                C2 = GB_spec_select (Cin, [ ], [ ], op, A, k, [ ]) ;
                GB_spec_compare (C1, C2) ;
            end
        end
    end
end

% full case
A.matrix = sparse (ones (n,n)) ;
for k2 = 1:length (select_ops)
    op = select_ops {k2} ;
    fprintf ('%s ', op) ;
    for s = 8
        A.sparsity = s ;
        for k1 = 1:length (types)
            type = types {k1} ;
            A.class = type ;
            if (test_contains (type, 'complex'))
                continue ;
            end
            for Thunk = -1:1
                k = sparse (Thunk) ;
                C1 = GB_mex_select  (Cin, [ ], [ ], op, A, k, [ ]) ;
                C2 = GB_spec_select (Cin, [ ], [ ], op, A, k, [ ]) ;
                GB_spec_compare (C1, C2) ;
            end
        end
    end
end

% test iso resize
fprintf ('resize\n') ;
C1 = GB_spec_resize (A, 5, 15) ;
C2 = GB_mex_resize  (A, 5, 15) ;
GB_spec_compare (C1, C2) ;

A.class = 'double complex' ;
GB_builtin_complex_set (true) ;
C1 = GB_spec_resize (A, 5, 15) ;
C2 = GB_mex_resize  (A, 5, 15) ;
GB_spec_compare (C1, C2) ;
GB_builtin_complex_set (false) ;

% iso select with user selectop
A.class = 'double' ;
C1 = GB_mex_band (A, -2, 2) ;
C2 = triu (tril (A.matrix,2), -2) ;
C3 = GB_mex_band2 (A, -2, 2) ;
assert (isequal (C1, C2)) ;
assert (isequal (C1, C3)) ;

% non-iso select with user selectop (see also test27)
B = A.matrix ;
C1 = GB_mex_band (B, -2, 2) ;
C2 = triu (tril (B,2), -2) ;
C3 = GB_mex_band (B, -2, 2) ;
assert (isequal (C1, C2)) ;
assert (isequal (C1, C3)) ;

GB_mex_burble (0) ;
GB_builtin_complex_set (true) ;
fprintf ('\ntest206: all tests passed\n') ;

