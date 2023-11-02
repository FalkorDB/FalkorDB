% cats_demo.m

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

clear

for scale = 1:6

    n = 60 * 10^scale ;
    nz = 25 * n ;
    d = nz / n^2 ;
    fprintf ('\n:::::: n = %g  nz = %g\n', n, nz) ;

    % create a random sparse matrix:
    rng ('default') ;
    tic
    A1 = sprand (n, n, d) ;
    t1 = toc ;
    fprintf ('create builtin sprand: %g sec\n', t1) ;

    % or create it with GrB.random: same pattern, but different
    % values because of how duplicates are handled
    if (demo_octave)
        A2 = GrB (A1) ;
    else
        rng ('default') ;
        tic
        A2 = GrB.random (n, n, d) ;
        t2 = toc ;
        assert (isequal (spones (A1), spones (A2)))
        fprintf ('create @GrB    sprand: %g sec\n', t2) ;
        fprintf ('@GrB speedup: %g\n', t1/t2) ;
        % make them identical
        A1 = double (A2) ;
    end

    GrB.burble (1) ;
    tic ; C1 = [A1 A1] ; t1 = toc ;      % using built-in sparse matrices
    tic ; C2 = [A2 A2] ; t2 = toc ;      % using @GrB sparse matrices
    GrB.burble (0) ;
    assert (isequal (C1, C2)) ;
    fprintf ('\n') ;
    fprintf ('builtin C = [A A] : %g sec\n', t1) ;
    fprintf ('@GrB    C = [A A] : %g sec\n', t2) ;
    fprintf ('@GrB speedup: %g\n', t1/t2) ;
    clear C1 C2 A2

    S = cell (2,2) ;
    for k = 1:4
        S {k} = A1 ;    % using a built-in sparse matrix
    end

    % the matrix gets too big at higher scales to have both C1 and C2
    % in memory at the same time
    GrB.burble (1) ;
    if (scale > 5)
        tic ; C = cell2mat (S)     ; t1 = toc ;
        clear C
        tic ; C = GrB.cell2mat (S) ; t2 = toc ;
        clear C
        GrB.burble (0) ;
    else
        tic ; C1 = cell2mat (S)     ; t1 = toc ;
        tic ; C2 = GrB.cell2mat (S) ; t2 = toc ;
        GrB.burble (0) ;
        assert (isequal (C1, C2)) ;
    end

    fprintf ('\n') ;
    fprintf ('builtin C = cell2mat (S)     : %g sec\n', t1) ;
    fprintf ('@GrB    C = GrB.cell2mat (S) : %g sec\n', t2) ;
    fprintf ('@GrB speedup: %g\n', t1/t2) ;
end

