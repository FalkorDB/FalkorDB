function test164
%TEST164 test dot method

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

n = 10 ;
A = GB_spec_random (n, n, 0.5, 1, 'logical') ;
A.sparsity = 2 ;    % sparse
B = GB_spec_random (n, 1, 0.5, 1, 'logical') ;
B.sparsity = 4 ;    % bitmap

semiring.add = 'any' ;
semiring.multiply = 'secondi1' ;
semiring.class = 'int32' ;

M = GB_spec_random (n, 1, inf, 1, 'int32') ;
M.matrix = double (full (M.matrix > 0.5)) ;
M.sparsity = 8 ;    % full
desc = struct ('inp0', 'tran', 'mask', 'complement') ;

% no accum
Cin = sparse (n, 1) ;

% can't compare C0 and C1 because the ANY monoid differs
C0 = GB_spec_mxm (Cin, M, [ ], semiring, A, B, desc) ;
C1 = GB_mex_mxm  (Cin, M, [ ], semiring, A, B, desc) ;
% GB_spec_compare (C0,C1) ;

semiring.add = 'min' ;
C0 = GB_spec_mxm (Cin, M, [ ], semiring, A, B, desc) ;
C1 = GB_mex_mxm  (Cin, M, [ ], semiring, A, B, desc) ;
GB_spec_compare (C0,C1) ;

fprintf ('test164: all tests passed\n') ;
