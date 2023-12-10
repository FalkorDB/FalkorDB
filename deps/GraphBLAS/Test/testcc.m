function testcc(use_builtin)
%TESTCC test complex transpose

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin < 1)
    use_builtin = true ;
end
GB_builtin_complex_set (use_builtin) ;

dt = struct ('inp0', 'tran') ;
seed = 1 ;
for m = [1 5 10 100]
    for n = [1 5 10 100]
        for trial = 1:100

            A = GB_mex_random (m, n, 10*(m+n), 1, seed) ; seed = seed + 1 ;
            S = GB_mex_complex (sparse (n,m)) ;
            B = GB_mex_random (n, m, 10*(m+n), 1, seed) ; seed = seed + 1 ;
            D = GB_mex_random (m, n, 10*(m+n), 1, seed) ; seed = seed + 1 ;
            C1 = A.' ;
            C2 = GB_mex_transpose (S, [], [], A, []) ;
            assert (isequal (C1, C2.matrix)) ;
            C1 = B + A.' ;
            C2 = GB_mex_transpose (B, [], 'plus', A, []) ;
            assert (isequal (C1, C2.matrix)) ;
            C1 = D + A ;
            C2 = GB_mex_transpose (D, [], 'plus', A, dt) ;
            assert (isequal (C1, C2.matrix)) ;

        end
    end
end

fprintf ('testcc: all complex transpose tests passed\n') ;

GB_builtin_complex_set (true) ;
