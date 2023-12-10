function dobench_methods (matrix, what)
% dobench_methods (matrix, what)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

Prob = ssget (matrix) ;
A = Prob.A ;
[m n] = size (A) ;
anz = nnz (A) ;
fprintf ('\n%s: n: %g million nnz: %g million\n', Prob.name, ...
    n/1e6, anz/1e6) ;
clear Prob ;
G = GrB (A) ;

if (nargin < 2)
    what = 4 ;
end

ntrials = 100 ;
tm = zeros (ntrials,1) ;
tg = zeros (ntrials,1) ;

if (what >= 1)

    % test y = S*x (mv)
    x = rand (n,1) ;
    ktrials = ntrials ;
    for k = 1:ntrials
        tic ; y = A*x ; tm (k) = toc ;
        tic ; z = G*x ; tg (k) = toc ;
        if (long_enough (tm, tg, k))
            ktrials = k ;
            break ;
        end
    end
    [tM tG] = summary (tm, tg, ktrials) ;
    fprintf ('y=S*x:   MAT: %8.4f GrB: %8.4f speedup: %8.2f\n', tM, tG, tM/tG) ;
    err = norm (y-z,1) / norm (y,1) ;
    assert (err < 1e-9) ;
    clear y z

    % test y = x*S (mv)
    x = rand (1,m) ;
    ktrials = ntrials ;
    for k = 1:ntrials
        tic ; y = x*A ; tm (k) = toc ;
        tic ; z = x*G ; tg (k) = toc ;
        if (long_enough (tm, tg, k))
            ktrials = k ;
            break ;
        end
    end
    [tM tG] = summary (tm, tg, ktrials) ;
    fprintf ('y=x*S:   MAT: %8.4f GrB: %8.4f speedup: %8.2f\n', tM, tG, tM/tG) ;
    err = norm (y-z,1) / norm (y,1) ;
    assert (err < 1e-9) ;
    clear y z

    % test C = S*F (mm)
    x = rand (n,4) ;
    ktrials = ntrials ;
    for k = 1:ntrials
        tic ; y = A*x ; tm (k) = toc ;
        tic ; z = G*x ; tg (k) = toc ;
        if (long_enough (tm, tg, k))
            ktrials = k ;
            break ;
        end
    end
    [tM tG] = summary (tm, tg, ktrials) ;
    fprintf ('C=S*F:   MAT: %8.4f GrB: %8.4f speedup: %8.2f\n', tM, tG, tM/tG) ;
    err = norm (y-z,1) / norm (y,1) ;
    assert (err < 1e-9) ;
    clear y z

    % test C = F*S (mm)
    x = rand (4,m) ;
    ktrials = ntrials ;
    for k = 1:ntrials
        tic ; y = x*A ; tm (k) = toc ;
        tic ; z = x*G ; tg (k) = toc ;
        if (long_enough (tm, tg, k))
            ktrials = k ;
            break ;
        end
    end
    [tM tG] = summary (tm, tg, ktrials) ;
    fprintf ('C=F*S:   MAT: %8.4f GrB: %8.4f speedup: %8.2f\n', tM, tG, tM/tG) ;
    err = norm (y-z,1) / norm (y,1) ;
    assert (err < 1e-9) ;
    clear y z

    % test C = Left*A (sp2m)
    d = 1000 / (8*m) ;
    x = sprand (8,m,d) ;
    ktrials = ntrials ;
    for k = 1:ntrials
        tic ; y = x*A ; tm (k) = toc ;
        tic ; z = x*G ; tg (k) = toc ;
        if (long_enough (tm, tg, k))
            ktrials = k ;
            break ;
        end
    end
    [tM tG] = summary (tm, tg, ktrials) ;
    fprintf ('C=L*S:   MAT: %8.4f GrB: %8.4f speedup: %8.2f\n', tM, tG, tM/tG) ;
    err = norm (y-z,1) / norm (y,1) ;
    assert (err < 1e-9) ;
    clear y z

    % test C = A*Right (sp2m)
    d = 1000 / (8*n) ;
    x = sprand (n,8,d) ;
    ktrials = ntrials ;
    for k = 1:ntrials
        tic ; y = A*x ; tm (k) = toc ;
        tic ; z = G*x ; tg (k) = toc ;
        if (long_enough (tm, tg, k))
            ktrials = k ;
            break ;
        end
    end
    [tM tG] = summary (tm, tg, ktrials) ;
    fprintf ('C=S*R:   MAT: %8.4f GrB: %8.4f speedup: %8.2f\n', tM, tG, tM/tG) ;
    err = norm (y-z,1) / norm (y,1) ;
    assert (err < 1e-9) ;
    clear y z
end

if (what >= 2)
    % test C = A' (transpose)
    ktrials = ntrials ;
    for k = 1:ntrials
        tic ; y = A' ; tm (k) = toc ;
        tic ; z = G' ; tg (k) = toc ;
        if (long_enough (tm, tg, k))
            ktrials = k ;
            break ;
        end
    end
    [tM tG] = summary (tm, tg, ktrials) ;
    fprintf ('C=S''     MAT: %8.4f GrB: %8.4f speedup: %8.2f\n', tM, tG, tM/tG) ;
    err = norm (y-z,1) / norm (y,1) ;
    assert (err < 1e-9) ;
    clear y z
end

if (what >= 3)

    % test C = A+B (add)
    ktrials = ntrials ;
    for k = 1:ntrials
        tic ; y = A+A ; tm (k) = toc ;
        tic ; z = G+G ; tg (k) = toc ;
        if (long_enough (tm, tg, k))
            ktrials = k ;
            break ;
        end
    end
    [tM tG] = summary (tm, tg, ktrials) ;
    fprintf ('C=S+S:   MAT: %8.4f GrB: %8.4f speedup: %8.2f\n', tM, tG, tM/tG) ;
    err = norm (y-z,1) / norm (y,1) ;
    assert (err < 1e-9) ;
    clear y z

    % test C = A+B (add)
    d = anz / (n*m) ;
    x = sprand (m,n,d/10) ;
    ktrials = ntrials ;
    for k = 1:ntrials
        tic ; y = A+x ; tm (k) = toc ;
        tic ; z = G+x ; tg (k) = toc ;
        if (long_enough (tm, tg, k))
            ktrials = k ;
            break ;
        end
    end
    [tM tG] = summary (tm, tg, ktrials) ;
    fprintf ('C=S+B:   MAT: %8.4f GrB: %8.4f speedup: %8.2f\n', tM, tG, tM/tG) ;
    err = norm (y-z,1) / norm (y,1) ;
    assert (err < 1e-9) ;
    clear y z

    % test C = A(p,p)
    p = randperm (m) ;
    q = randperm (n) ;
    ktrials = ntrials ;
    for k = 1:ntrials
        tic ; y = A (p,q) ; tm (k) = toc ;
        tic ; z = G (p,q) ; tg (k) = toc ;
        if (long_enough (tm, tg, k))
            ktrials = k ;
            break ;
        end
    end
    [tM tG] = summary (tm, tg, ktrials) ;
    fprintf ('C=S(p,q) MAT: %8.4f GrB: %8.4f speedup: %8.2f\n', tM, tG, tM/tG) ;
    err = norm (y-z,1) / norm (y,1) ;
    assert (err < 1e-9) ;
    clear y z

end

if (what >= 4)

    % test C = A^2 (square)
    if (m == n)

        ktrials = ntrials ;
        for k = 1:ntrials
            tic ; y = A^2 ; tm (k) = toc ;
            tic ; z = G^2 ; tg (k) = toc ;
            if (long_enough (tm, tg, k))
                ktrials = k ;
                break ;
            end
        end
        [tM tG] = summary (tm, tg, ktrials) ;
        fprintf ('C=S^2:   MAT: %8.4f GrB: %8.4f speedup: %8.2f\n', tM, tG, tM/tG) ;
        err = norm (y-z,1) / norm (y,1) ;
        assert (err < 1e-9) ;
        clear y z

    else

        ktrials = ntrials ;
        for k = 1:ntrials
            tic ; y = A'*A ; tm (k) = toc ;
            tic ; z = G'*G ; tg (k) = toc ;
            if (long_enough (tm, tg, k))
                ktrials = k ;
                break ;
            end
        end
        [tM tG] = summary (tm, tg, ktrials) ;
        fprintf ('C=S''*S   MAT: %8.4f GrB: %8.4f speedup: %8.2f\n', tM, tG, tM/tG) ;
        err = norm (y-z,1) / norm (y,1) ;
        assert (err < 1e-9) ;
        clear y z

    end
end


