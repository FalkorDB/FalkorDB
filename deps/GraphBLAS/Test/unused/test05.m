function test05
%TEST05 test GrB_*_setElement

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

A = sparse (5,5) ;
A (2,2) = 42 ;

I = [1 2 3 2]' ;
J = [1 2 3 2]' ;

I0 = uint64 (I-1) ;
J0 = uint64 (I-1) ;
X = [4 5 6 3]' ;


C1 = GB_mex_setElement (A, I0, J0, int8(X))  ;
C2 = GB_mex_setElement (A, I0, J0, int8(X), false, true)  ;

B = A ;
for k = 1:length(I)
    B (I (k), J (k)) = X (k) ;
end

assert (isequal (C1.matrix, B)) 
assert (isequal (C2.matrix, B)) 

A2.matrix = A ;

for A_is_hyper = 0:1
    for A_is_csc = 0:1
        A2.is_csc   = A_is_csc ;
        A2.is_hyper = A_is_hyper ;

        C3 = GB_mex_setElement (A2, I0, J0, int8(X))  ;
        C4 = GB_mex_setElement (A2, I0, J0, int8(X), false, true)  ;
        assert (isequal (C3.matrix, B)) 
        assert (isequal (C4.matrix, B)) 

    end
end

fprintf ('\ntest05: all tests passed\n') ;

