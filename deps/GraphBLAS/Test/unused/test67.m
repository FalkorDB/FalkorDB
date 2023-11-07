function test67
%TEST67 test GrB_apply

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('\n---------------------------- quick test of GrB_apply\n') ;

A = sparse (rand (4,3)) ;
C = sparse (rand (4,3)) ;

C0 = GB_mex_apply (C, [ ], '', 'ainv', A) ;
C1 = -A ;
assert (isequal (full (C0.matrix), C1))

C0 = GB_mex_apply (C, [ ], 'plus', 'identity', A) ;
C1 = C + A ;
assert (isequal (full (C0.matrix), C1))

C0 = GB_mex_apply (C, [ ], 'times', 'minv', A) ;
C1 = C .* (1./ A) ;
assert (isequal (full (C0.matrix), C1))

d = struct ('inp0', 'tran') ;
C0 = GB_mex_apply (C', [ ], 'times', 'ainv', A, d) ;
C1 = C' .* (-A') ;
assert (isequal (full (C0.matrix), C1))

A = sparse (zeros (4,3)) ;
C0 = GB_mex_apply (A', [ ], 'plus', 'ainv', A, d) ;
assert (nnz (C0.matrix) == 0)
C0 = GB_mex_apply (A, [ ], '', 'ainv', A) ;
assert (nnz (C0.matrix) == 0)

fprintf ('\ntest67: all tests passed\n') ;

