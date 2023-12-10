function test140
%TEST140 test assign with duplicates

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

n = 6 ;
A = 100 * sprand (n, n, 0.5) ;
M = sparse (rand (n)) > 0.5 ;
Cin = sprand (n, n, 0.5) ;

Cout = gb.assign (Cin, A, { }, { }) ;
assert (isequal (A, sparse (Cout))) ;

I = [2 1 5] ;
J = [3 3 1 2] ;
% J = [2 2 1 3] ;
B = sprandn (length (I), length (J), 0.5) ;

    [J1 J1k] = sort (J) 
    J1k = J1k-1 
    Jduplicate = [(J1 (1:end-1) == J1 (2:end)), false] 
    J2  = J1  (~Jduplicate) 
    J2k = J1k (~Jduplicate) 

I0 = uint64(I)-1 ;
J0 = uint64(J)-1 ;

C_spec = GB_spec_assign (Cin, [ ], [ ], B, I, J, [ ], [ ])
C_mex  = GB_mex_assign  (Cin, [ ], [ ], B, I0, J0, [ ], [ ])
GB_spec_compare (C_spec, C_mex) ;

