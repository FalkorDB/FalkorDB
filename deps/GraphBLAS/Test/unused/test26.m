function test26(longtests)
%TEST26 performance test for GxB_select

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('\ntest26 ------------------------------performance of GxB_select\n') ;

[save_nthreads save_chunk] = nthreads_get ;
chunk = 4096 ;
nthreads = feature_numcores ;
nthreads_set (nthreads, chunk) ;

[~, ~, ~, ~, ~, select_ops] = GB_spec_opsall ;

if (nargin < 1)
    longtests = 0 ;
end

if (longtests)
    % ssget will be used
    nprobs = 5 ;
else
    nprobs = 3 ;
end

rng ('default') ;

dt = struct ('inp0', 'tran') ;

for probs = 1:nprobs

    switch probs
        case 1
            A = sparse (200, 1, 0.1) ;
        case 2
            A = sparse (200, 100, 0.1) ;
        case 3
            A1 = spones (sprand (10, 10, 0.5)) ;
            A2 = spones (sprand (10, 10, 0.5)) ;
            A3 = sparse (10,10) ;
            A4 = GB_mex_Matrix_eWiseAdd (A3, [], [], 'minus', A1, A2, [ ]) ;
            A = A4.matrix ;
            % GB_spok(A) will fail since it has intentional explicit zeros
        case 4
            A = sparse (rand (6000)) ;
        case 5
            Prob = ssget (2662) ;
            A = Prob.A ;
    end

    [m n] = size (A) ;
    fprintf ('\nProblem: m %d n %d nnz %d\n', m, n, nnz (A)) ;
    Cin = sparse (m,n) ;

    for k2 = 1:length(select_ops)
        op = select_ops {k2} ;
        fprintf ('%s:\n', op) ;

        for k = [-m -floor(m/2) -50 -1 0 1 50 floor(n/2) n]

            fprintf ('k: %10d ', k) ;

            tic
            C1 = GB_mex_select (Cin, [], [], op, A, k, []) ;
            t1 = toc ;
            fprintf ('GB: %10.6f ', t1) ;

            C3 = 'none' ;
            C2 = 'none' ;

            switch (op)
                case 'tril'
                    tic
                    C2 = tril (A,k) ;
                    t2 = toc ;
                    tic
                    C3 = GB_mex_tril (A, k) ;
                    t3 = toc ;
                case 'triu'
                    tic
                    C2 = triu (A,k) ;
                    t2 = toc ;
                    tic
                    C3 = GB_mex_triu (A, k) ;
                    t3 = toc ;
                case 'diag'
                    if (size (A,2) > 1)
                        tic
                        C2 = spdiags (spdiags (A,k), k, m, n) ;
                        t2 = toc ;
                    end
                    tic
                    C3 = GB_mex_diag (A, k) ;
                    t3 = toc ;
                case 'offdiag'
                    if (size (A,2) > 1)
                        tic
                        C2 = A - spdiags (spdiags (A,k), k, m, n) ;
                        t2 = toc ;
                    end
                    tic
                    C3 = GB_mex_offdiag (A, k) ;
                    t3 = toc ;
                case 'nonzero'
                    tic
                    C2 = A .* (A ~= 0) ;
                    t2 = toc ;
                    assert (isequal (1*C2,1*A)) ;
                    tic
                    C3 = GB_mex_nonzero (A) ;
                    t3 = toc ;
                    assert (isequal (1*C3,1*A)) ;
            end

            if (~ischar (C3))
                fprintf (' %10.6f ', t3) ;
                assert (isequal (C3, C1.matrix))
            end

            if (~ischar (C2))
                fprintf ('builtin: %10.6f ', t2) ;
                fprintf ('nnz: %10d speedup %5.2f ', nnz (C2), t2/t1) ;

                if (~ischar (C3))
                    fprintf (' %5.2f ', t2/t3) ;
                end
                assert (isequal (1*C1.matrix, 1*C2))
            end
            fprintf ('\n') ;

        end
    end
end

ok = true ;
A = sparse (ones (4)) ;
try
    C = GB_mex_select (A, [ ], [ ], 'tril', A, A, [ ]) ;
    ok = false ;
catch me
    fprintf ('\nexpected error: %s\n', me.message) ;
end

nthreads_set (save_nthreads, save_chunk) ;
fprintf ('test26: all tests passed\n') ;
