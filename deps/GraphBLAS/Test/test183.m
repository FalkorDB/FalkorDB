function test183
%TEST183 test GrB_eWiseMult with a hypersparse mask 

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[~, ~, ~, types, ~, ~] = GB_spec_opsall ;
types = types.all ;

fprintf ('test183 -----------eWiseMult with hypersparse mask\n') ;

n = 10 ;

rng ('default') ;
dtt = struct ( 'inp0', 'tran', 'inp1', 'tran' ) ;

op.opname = 'times' ;
op.optype = 'double' ;
tol = 1e-12 ;

% for trial = 1:5

    A = GB_spec_random (n, n, inf, 1, 'double') ;
    A.matrix (1,1) = 0 ;
    A.pattern = logical (A.matrix) ;
    A.pattern (1,1) = false ;
    A.sparsity = 2 ;    % sparse
    B = GB_spec_random (n, n, inf, 1, 'double') ;
    B.sparsity = 2 ;    % sparse
    M = GB_spec_random (n, n, 0.001, 1, 'logical') ;
    M.sparsity = 1 ;    % hypersparse
    C = sparse (n, n) ;

    %---------------------------------------
    % A'.*B', with mask
    %---------------------------------------

    C0 = GB_spec_Matrix_eWiseMult (C, M, [ ], op, A, B, dtt) ;
    C1 = GB_mex_Matrix_eWiseMult  (C, M, [ ], op, A, B, dtt) ;
    GB_spec_compare (C0, C1, 0, tol) ;

% end

fprintf ('\ntest183: all tests passed\n') ;

