function test246 (dohack)
%TEST246 test GrB_mxm (for GB_AxB_saxpy3_fineHash_phase2.c)
%
% This tests the "if (hf == i_unlocked) // f == 2" case in the last block of
% code in GB_AxB_saxpy3_fineHash_phase2.c.  The test is nondeterministic so
% it the test coverage might vary, or even be zero.  See also test247.m.
% It is thus run for many trials below.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('test246: testing of GB_AxB_saxpy3_fineHash_phase2.c\n') ;

rng ('default') ;

n = 1000 ;

A = sprand (n, n, 0.5) ;

% save current global settings
save = GB_mex_hack ;
[nthreads_save chunk_save] = nthreads_get ;

% modify the global settings
hack = save ;
if (nargin < 1)
    dohack = 2 ;        % use 0 for default, 2 for prior setting in v7.2.0
end
hack (1) = dohack ; GB_mex_hack (hack) ; % "very_costly" in saxpy3

semiring.multiply = 'times' ;
semiring.add = 'plus' ;
semiring.class = 'double' ;

for k = [1 2 4 16 128]
    S = sparse (n, k) ;
    M = spones (sprand (n, k, 0.1)) ;

    for da = [0.001 0.5]
        A = sprand (n, n, da) ;
        H.matrix = A ;
        H.sparsity = 1 ;

        for db = [0 0.01, 0.5]
            if (db == 0 && k == 4)
                B = sprand (n, k, 0.01) ;
                B (:,2) = pi ;
                B (1:(n/10),3) = 3 ;
            elseif (db == 0 && k > 1)
                B = sprand (n, k, 0.01) ;
                B (:,2) = pi ;
            else
                B = sprand (n, k, db) ;
            end

            for method = [0 1 2]
                if (method == 0)
                    desc = [ ] ;
                elseif (method == 1)
                    desc.axb = 'gustavson' ;
                else
                    desc.axb = 'hash' ;
                end

                for threads = [1 4 16]
                    nthreads_set (threads, 1) ;
                    fprintf ('.') ;

                    for trial = 1:5

                    % no mask
                    C1 = A*B ;
                    C2 = GB_mex_mxm (S, [ ], [ ], semiring, A, B, desc) ;
                    err = norm (C1 - C2.matrix, 1) / max (1, norm (C1, 1)) ;
                    assert (err < 1e-12)

                    % with the mask
                    C1 = M.*(A*B) ;
                    C2 = GB_mex_mxm (S, M, [ ], semiring, A, B, desc) ;
                    err = norm (C1 - C2.matrix, 1) / max (1, norm (C1, 1)) ;
                    assert (err < 1e-12)

                    % hypersparse case
                    C2 = GB_mex_mxm (S, M, [ ], semiring, H, B, desc) ;
                    err = norm (C1 - C2.matrix, 1) / max (1, norm (C1, 1)) ;
                    assert (err < 1e-12)

                    end
                end
            end
        end
    end
end

% restore global settings
nthreads_set (nthreads_save, chunk_save) ;
GB_mex_hack (save) ;

fprintf ('\ntest246: all tests passed\n') ;

