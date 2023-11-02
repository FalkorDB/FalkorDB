function test58 (cover)
%TEST58 test GrB_eWiseAdd

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin < 1)
    cover = 1 ;
end

fprintf ('\ntest58: ----- quick performance for GB_mex_Matrix_eWiseAdd\n') ;

[save save_chunk] = nthreads_get ;
chunk = 4096 ;
nthreads = feature_numcores ;
nthreads_set (nthreads, chunk) ;


add = 'plus' ;
Mask = [ ] ;
accum = 'plus' ;

if (~cover)
    Prob = ssget (939) ;
    A = Prob.A ;
    B = 2*A ;
    Cin = A .* (A < 0) ;

    tic
    C = Cin + (A+B) ;
    t1 =toc ;

    tic
    C2 = GB_mex_Matrix_eWiseAdd (Cin, Mask, accum, add, A, B, [ ]) ;
    t2 = toc ;
    assert (isequal (C2.matrix,  C))

    fprintf ('built-in: %g GB: %g  speedup: %g\n', t1, t2, t1/t2) ;
end

if (cover)
    nn = [10 100 ] ;
else
    nn = [10 100 1000 10000 50000] ;
end

for m = nn
    for n = nn
        A = sprandn (m, n, 0.01) ;
        B = sprandn (m, n, 0.01) ;
        Cin = sprandn (m, n, 0.01) ;
        AT = A' ;
        BT = B' ;

        if (min (m,n) >= 10000)
            trials = 1 ;
        else
            trials = 20 ;
        end

        Dnn = struct ;
        Dtn = struct ('inp0', 'tran') ;
        Dnt = struct ('inp1', 'tran') ;
        Dtt = struct ('inp0', 'tran', 'inp1', 'tran') ;

        % C += A+B
        tic
        for k = 1:trials
            C1 = Cin + (A+B) ;
        end
        t1 = toc / trials ;

        tic
        for k = 1:trials
            C2 = GB_mex_Matrix_eWiseAdd (Cin, [ ], accum, add, A, B, [ ]) ;
        end
        tg = toc ;
        t2 = tg /trials ;
        assert (isequal (C1, C2.matrix)) ;

        fprintf ('A+B:   ') ;
        fprintf ('m %6d n %6d nz %8d: built-in %8.4f GrB %8.4f', ...
            m, n, nnz (C1), t1, t2) ;
        % fprintf (' Cs: %8.4f', t3) ;
        fprintf (' speedup %g\n', t1/t2) ;

        % C += A+B'
        tic
        for k = 1:trials
            C1 = Cin + (A+BT') ;
        end
        t1 = toc / trials ;

        tic ;
        for k = 1:trials
            C2 = GB_mex_Matrix_eWiseAdd (Cin, [ ], accum, add, A, BT, Dnt) ;
        end
        tg = toc ;
        t2 = tg /trials ;
        assert (isequal (C1, C2.matrix)) ;

        fprintf ('A+B'':  ') ;
        fprintf ('m %6d n %6d nz %8d: built-in %8.4f GrB %8.4f', ...
            m, n, nnz (C1), t1, t2) ;
        % fprintf (' Cs: %8.4f', t3) ;
        fprintf (' speedup %g\n', t1/t2) ;

        % C += A'+B
        tic
        for k = 1:trials
            C1 = Cin + (AT'+B) ;
        end
        t1 = toc / trials  ;

        tic
        for k = 1:trials
            C2 = GB_mex_Matrix_eWiseAdd (Cin, [ ], accum, add, AT, B, Dtn) ;
        end
        tg = toc ;
        t2 = tg /trials ;
        assert (isequal (C1, C2.matrix)) ;

        fprintf ('A''+B:  ') ;
        fprintf ('m %6d n %6d nz %8d: built-in %8.4f GrB %8.4f', ...
            m, n, nnz (C1), t1, t2) ;
        % fprintf (' Cs: %8.4f', t3) ;
        fprintf (' speedup %g\n', t1/t2) ;

        % C += A'+B'
        tic
        for k = 1:trials
            C1 = Cin + (AT'+BT') ;
        end
        t1 = toc / trials ;

        tic ;
        for k = 1:trials
            C2 = GB_mex_Matrix_eWiseAdd (Cin, [ ], accum, add, AT, BT, Dtt) ;
        end
        tg = toc ;
        t2 = tg /trials ;
        assert (isequal (C1, C2.matrix)) ;

        fprintf ('A''+B'': ') ;
        fprintf ('m %6d n %6d nz %8d: built-in %8.4f GrB %8.4f', ...
            m, n, nnz (C1), t1, t2) ;
        % fprintf (' Cs: %8.4f', t3) ;
        fprintf (' speedup %g\n', t1/t2) ;

    end
end

fprintf ('\ntest58: all tests passed\n') ;

nthreads_set (save, save_chunk) ;
