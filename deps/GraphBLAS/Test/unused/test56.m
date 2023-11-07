function test56
%TEST56 test GrB_*_build

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

op.opname = 'min' ;
op.optype = 'uint32' ;

fprintf ('ktriplets:\n') ;
I = uint64 ([0 0 0]') ;
J = uint64 ([0 0 0]') ;
Xk = int16 ([-93 74 -85]') 

fprintf ('ptriplets:\n') ;
Xp = int16 ([-93 -85 74]') ;

A1 = GB_mex_Matrix_build (I, J, Xk, 1, 1, op);
A1_matrix = full (double (A1.matrix))

A2 = GB_mex_Matrix_build (I, J, Xp, 1, 1, op);
A2_matrix = full (double (A2.matrix))

S1 = GB_spec_build       (I, J, Xk, 1, 1, op);
S1_matrix = double (S1.matrix)

S2 = GB_spec_build       (I, J, Xp, 1, 1, op);
S2_matrix = double (S2.matrix)

fprintf ('\ntest56: all tests passed\n') ;

