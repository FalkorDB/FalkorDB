function testc5(use_builtin)
%TESTC5 test complex subref

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin < 1)
    use_builtin = true ;
end
GB_builtin_complex_set (use_builtin) ;

seed = 1 ;
for m = [1 5 10 100]
    for n = [1 5 10 100]
        seed = seed + 1 ;
        A = GB_mex_random (m, n, 10*(m+n), 1, seed) ;
        for trials = 1:10

            J = randperm (n, 1+floor(n/2)) ;
            I = randperm (m, 1+floor(m/2)) ;
            J0 = uint64 (J-1) ;
            I0 = uint64 (I-1) ;

            C1 = GB_mex_Matrix_subref (A, I0, J0) ;
            C2 = GB_mex_complex (A (I,J)) ;
            assert (isequal (C1, C2))
        end
    end
end
fprintf ('testc5: all complex subref C = A(I,J) tests passed\n') ;

GB_builtin_complex_set (true) ;
