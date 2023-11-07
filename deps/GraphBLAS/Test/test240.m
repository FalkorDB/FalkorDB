function test240
%TEST240 test GrB_mxm: dot4, saxpy4, saxpy5

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[binops, ~, add_ops, types, ~, ~] = GB_spec_opsall ;
mult_ops = binops.all ;
types = types.all ;

fprintf ('test240 -------- GrB_mxm dot4 and saxpy5\n') ;

rng ('default') ;
GB_builtin_complex_set (true) ;

dnn = struct ;
dtn = struct ( 'inp0', 'tran' ) ;
dnt = struct ( 'inp1', 'tran' ) ;
dtt = struct ( 'inp0', 'tran', 'inp1', 'tran' ) ;

monoid.opname = 'plus' ;
monoid.optype = 'double' ;
semiring.add = monoid ;
semiring.multiply = 'times' ;
semiring.class = 'double' ;

n = 200 ;
is_csc = true ;
A = GB_spec_random (n, n, 0.5, 100, 'double', is_csc) ;
A.sparsity = 2 ;    % A is sparse

% test dot4
for k = 1:32
    % C += A'*B
    B = rand (n, k) ;
    F = rand (n, k) ;
    C1 = GB_mex_mxm_update (F, semiring, A, B, dtn) ;
    C2 = F + A.matrix'*B ;
    GB_spec_compare (C1, C2, 0, 1e-12) ;
end

% test saxpy5: A full, B sparse
for k = 1:32
    % C += A*B
    B = rand (k, n) ;
    F = rand (k, n) ;
    C1 = GB_mex_mxm_update (F, semiring, B, A, [ ]) ;
    C2 = F + B*A.matrix ;
    GB_spec_compare (C1, C2, 0, 1e-12) ;
end

% test saxpy5: A full, B sparse, with typecasting
A2 = A ;
A2.class = 'single' ;
for k = 1:32
    % C += A*B
    B = rand (k, n) ;
    F = rand (k, n) ;
    C1 = GB_mex_mxm_update (F, semiring, B, A2, [ ]) ;
    C2 = F + B*A.matrix ;
    GB_spec_compare (C1, C2, 0, 1e-5) ;
end

% test saxpy5: A iso bitmap, B sparse
A.sparsity = 4 ;    % A is bitmap
A.iso = true ;      % A is bitmap
A.matrix = pi * spones (A.matrix) ;
for k = 1:32
    % C += A*B
    B = GB_spec_random (n, k, 0.5, 100, 'double', is_csc) ;
    B.sparsity = 2 ;    % B is sparse
    F = rand (n, k) ;
    C1 = GB_mex_mxm_update (F, semiring, A, B, [ ]) ;
    C2 = F + A.matrix * B.matrix ;
    GB_spec_compare (C1, C2, 0, 1e-12) ;
end

% GB_mex_burble (0) ;
fprintf ('\ntest240: all tests passed\n') ;

