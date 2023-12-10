function test186 (dohack)
%TEST186 test saxpy for all sparsity formats

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('test186 --------------- C<!M>A*B for all sparsity formats\n') ;

rng ('default') ;

% save current global settings, then modify them
save = GB_mex_hack ;
hack = save ;
if (nargin < 1)
    dohack = 2 ;
end
hack (1) = dohack ;     % modify "very_costly" in GxB_AxB_saxpy3_slice_balanced
GB_mex_hack (hack) ;

load west0479 ;
A.matrix = west0479 ;
A.class = 'double' ;
A.pattern = logical (spones (A.matrix)) ;
m = size (A.matrix, 1) ;

semiring.add = 'plus' ;
semiring.multiply = 'times' ;
semiring.class = 'double' ;

any_pair.add = 'any' ;
any_pair.multiply = 'pair' ;    % same as oneb
any_pair.class = 'double' ;

C0 = sparse (m, 1) ;
maxerr = 0 ;

M = sparse (rand (m, 1) > 0.5) ;
desc.mask = 'complement' ;

B = GB_spec_random (m, 1, 0.5, 1, 'double') ;
B2 = B ;
B2.class = 'single' ;

% using fine atomic tasks when A is sparse and B is bitmap
for A_sparsity = [1 2 4 8]
    for B_sparsity = [1 2 4 8]
        A.sparsity = A_sparsity ;
        B.sparsity = B_sparsity ;
        B2.sparsity = B_sparsity ;

        % C2<!M> = A*B using the conventional semiring
        C3 = double (~M) .* (A.matrix * B.matrix) ;
        C2 = GB_mex_mxm  (C0, M, [ ], semiring, A, B, desc) ;
        err = norm (C3 - C2.matrix, 1) / norm (C3, 1) ;
        maxerr = max (maxerr, err) ;
        assert (err < 1e-12) ;

        % C2<!M> = A*single(B) to force typecasting
        C1 = GB_mex_mxm  (C0, M, [ ], semiring, A, B2, desc) ;
        err = norm (C3 - C1.matrix, 1) / norm (C3, 1) ;
        maxerr = max (maxerr, err) ;
        assert (err < 1e-6) ;

        % C2<!M> = A*B using the any-pair semiring
        C3 = spones (C3) ;
        C2 = GB_mex_mxm  (C0, M, [ ], any_pair, A, B, desc) ;
        err = norm (C3 - C2.matrix, 1) / norm (C3, 1) ;
        maxerr = max (maxerr, err) ;
        assert (err < 1e-12) ;
    end
end

B3 = GB_spec_random (m, 3, 0.5, 1, 'double') ;
M3 = sparse (rand (m, 3) > 0.5) ;
C03 = sparse (m, 3)  ;

% using fine non-atomic tasks when A is sparse and B is bitmap
A.matrix = sprand (m, m, 0.8) ;
A.pattern = logical (spones (A.matrix)) ;
for A_sparsity = [1 2 4 8]
    for B_sparsity = [1 2 4 8]
        A.sparsity = A_sparsity ;
        B.sparsity = B_sparsity ;
        B3.sparsity = B_sparsity ;
        fprintf ('.') ;

        % C2<!M> = A*B using the conventional semiring
        C3 = double (~M) .* (A.matrix * B.matrix) ;
        C2 = GB_mex_mxm  (C0, M, [ ], semiring, A, B, desc) ;
        err = norm (C3 - C2.matrix, 1) / norm (C3, 1) ;
        maxerr = max (maxerr, err) ;
        assert (err < 1e-12) ;

        % C2<!M> = A*B using the any-pair semiring
        C3 = spones (C3) ;
        C2 = GB_mex_mxm  (C0, M, [ ], any_pair, A, B, desc) ;
        err = norm (C3 - C2.matrix, 1) / norm (C3, 1) ;
        maxerr = max (maxerr, err) ;
        assert (err < 1e-12) ;

        % C2<!M3> = A*B3 using the conventional semiring
        C3 = double (~M3) .* (A.matrix * B3.matrix) ;
        C2 = GB_mex_mxm  (C03, M3, [ ], semiring, A, B3, desc) ;
        err = norm (C3 - C2.matrix, 1) / norm (C3, 1) ;
        maxerr = max (maxerr, err) ;
        assert (err < 1e-12) ;
    end
end

% restore global settings
GB_mex_burble (0) ;
GB_mex_hack (save) ;

fprintf ('\n') ;
fprintf ('maxerr: %g\n', maxerr) ;
fprintf ('test186: all tests passed\n') ;

