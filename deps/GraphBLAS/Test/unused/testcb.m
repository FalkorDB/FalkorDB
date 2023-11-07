function testcb(use_builtin)
%TESTCB test complex reduce

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin < 1)
    use_builtin = true ;
end
GB_builtin_complex_set (use_builtin) ;

tol = 1e-13 ;
seed = 1 ;
for m = [1 5 10 100]
    for n = [1 5 10 100]
        for trial = 1:100

            A = GB_mex_random (m, n, 10*(m+n), 1, seed) ; seed = seed + 1 ;
            s = complex (0) ;

            c1 = complex (full (sum (A (:)))) ;
            c2 = GB_mex_reduce_to_scalar (s, [], 'plus', A) ;
            assert (abs (c1 - c2) <= tol * abs (c1) + tol) ;
            % assert (isequal (c1,c2))

            [i j x] = find (A) ;
            c1 = complex (prod (x)) ;
            c2 = GB_mex_reduce_to_scalar (s, [], 'times', A) ;
            assert (abs (c1 - c2) <= tol * abs (c1) + tol) ;
            % assert (isequal (c1,c2))

            s = complex (pi,2) ;
            c1 = s * complex (full (sum (A (:)))) ;
            c2 = GB_mex_reduce_to_scalar (s, 'times', 'plus', A) ;
            assert (abs (c1 - c2) <= tol * abs (c1) + tol) ;
            % assert (isequal (c1,c2))

            s = complex (pi,2) ;
            c1 = s + complex (full (sum (A (:)))) ;
            c2 = GB_mex_reduce_to_scalar (s, 'plus', 'plus', A) ;
            assert (abs (c1 - c2) <= tol * abs (c1) + tol) ;
            % assert (isequal (c1,c2))

            if (n > 1)

                w = GB_mex_complex (sparse (m, 1)) ;
                c1 = sum (A.').' ;
                c2 = GB_mex_reduce_to_vector (w, [], [], 'plus', A, []) ;
                assert (norm (c1 - c2.matrix, 1) <= tol * norm (c1, 1) + tol) ;
                % assert (isequal (c1,c2.matrix))

                w = GB_mex_random (m, 1, 10, 1, seed) ; seed = seed + 1 ;
                c1 = w + sum (A.').' ;
                c2 = GB_mex_reduce_to_vector (w, [], 'plus', 'plus', A, []) ;
                assert (norm (c1 - c2.matrix, 1) <= tol * norm (c1, 1) + tol) ;
                % assert (isequal (c1,c2.matrix))

            end
        end
    end
end

fprintf ('testcb: all complex reduce tests passed\n') ;

GB_builtin_complex_set (true) ;
