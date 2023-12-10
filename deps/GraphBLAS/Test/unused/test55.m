function test55
%TEST55 test GxB_subassign, illustrate duplicate indices

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

A = magic (5)

A ([2 2], [4 5])

B = [1000 800 ; 60000 2000 ]
% B (1,1) = nan

i = [2 2]
j = [4 5]

C1 = A ;
C1 ([2 2], [4 5]) = A ([2 2], [4 5]) + B

a = sparse (A)
b = sparse (B)
i0 = uint64 (i-1)
j0 = uint64 (j-1)

C2 = GB_mex_subassign (a, [], 'plus', b, i0, j0)

C1
full (C2.matrix)

c = sparse ([ 1 2 3 4 5 ])
i = uint64 (0)
j = uint64 ([ 0 0 4 ]) 
a = sparse ([ 10 100 1000]) 
e = GB_mex_subassign (c, [], 'plus', a, i, j)
e = full (e.matrix)

ac = accumarray (double (j+1)', full(a)')'
e2 = c ;
e2 (i+1, :) = e2 (i+1, :) + ac
e2 = full (e2)

e2 = c ;
e2 (i+1,j+1) = e2 (i+1,j+1) + a
e2 = full (e2)

a = sparse ([ 1000 1000 1000]) 
e = GB_mex_subassign (c, [], 'plus', a, i, j)
e = full (e.matrix)

e2 = c ;
e2 (i+1,j+1) = e2 (i+1,j+1) + a
e2 = full (e2)


e2 = c ;
e2 (i+1,j+1) = e2 (i+1,j+1) + 1000
e2 = full (e2)


a = sparse ([ 1000 1000 1000]) 
e = GB_mex_subassign (c, [], '', a, i, j)
e = full (e.matrix)

e2 = c ;
e2 (i+1,j+1) = a
e2 = full (e2)

e2 = c ;
e2 (i+1,j+1) = 1000 
e2 = full (e2)

fprintf ('\ntest55: all tests passed\n') ;

