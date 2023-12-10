function testc9(use_builtin)
%TESTC9 test complex extractTuples

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin < 1)
    use_builtin = true ;
end
GB_builtin_complex_set (use_builtin) ;

seed = 1 ;
for m = [1 5 10 100]
    for n = [1 5 10 100]
        for trial = 1:100
            A = GB_mex_random (m, n, 10*(m+n), 1, seed) ; seed = seed + 1 ;

            [I  J  X ] = find (A) ;
            X = complex (X) ;

            [I0 J0 X0] = GB_mex_extractTuples (A) ;

            I1 = double (I0+1) ;
            J1 = double (J0+1) ;

            assert (isequal (I (:), I1 (:))) ;
            assert (isequal (J (:), J1 (:))) ;
            assert (isequal (X (:), X0 (:))) ;

        end
    end
end

fprintf ('testc9: all complex extractTuples tests passed\n') ;

GB_builtin_complex_set (true) ;
