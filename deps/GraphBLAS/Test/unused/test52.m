function test52
%TEST52 test AdotB vs AxB

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('\n----------------------- AdotB versus AxB\n') ;

rng ('default')
x = sparse (rand (10,1)) ;
y = sparse (rand (10,1)) ;
c = x'*y ;
'AdotB'
C = GB_mex_AdotB (x, y)
'did AdotB'
c - C
assert (isequal (c, C))

x = sprandn (100, 1, 0.2) ;
y = sprandn (100, 1, 0.2) ;
c = x'*y ;
C = GB_mex_AdotB (x, y)
assert (isequal (c, C))

for m = 1:10
    for n = 1:10
        for k = [10 100 1000 10000]

            A = sprandn (k, m, 0.2) ;
            B = sprandn (k, n, 0.2) ;
            Mask = sprand (m, n, 0.5) ;

            C = A'*B ;
            C2 = GB_mex_AdotB (A,B) ;

            assert (isequal (C, C2)) ;
            assert (GB_spok (C2) == 1)

            C = spones (Mask) .* C ;
            C2 = GB_mex_AdotB (A,B, Mask) ;

            assert (isequal (C, C2)) ;
            assert (GB_spok (C2) == 1)
        end
    end
end

relwork = [ ] ;
reltime = [ ] ;

k = 10e6 ;
fprintf ('\nbuilding random sparse matrices %d by M\n', k) ;
for m = [1 10 20:10:60 61:65 70:10:100]
fprintf ('\n') ;
for n = [1 10 20:10:60 61:65 70:10:100]

    d = 0.001 ;
    A = sprandn (k, m, d) ;
    B = sprandn (k, n, d) ;
    Mask = spones (sprandn (m, n, 0.5)) ;
    % A (:,m) = sparse (rand (k,1)) ;
    % B (:,m) = sparse (rand (k,1)) ;

    cwork = m*n ;
    awork = min (nnz(A) + k + m, nnz(B) + k + n) ;

    relwork = [relwork cwork/awork] ;

    tic
    C = A'*B ;
    t1 = toc ;

    % fprintf ('GrB AdotB:\n') ;
    tic
    C2 = GB_mex_AdotB (A,B) ;
    t2 = toc ;

    % fprintf ('GrB A''*B native:\n') ;
    tic
    C4 = GB_mex_AxB (A,B, true) ;
    t4 = toc ;

    % fprintf ('GrB A''*B native:\n') ;
    tic
    C5 = GB_mex_AxB (A',B) ;
    t5 = toc ;

    reltime = [reltime t2/t5] ;

    fprintf (...
'm %3d n %3d %10.2e builtin: %10.4f AdotB : %10.4f GB,auto:: %10.4f outer %10.4f', ...
    m, n, cwork/awork, t1, t2, t4, t5) ;
    % fprintf (' speedup: %10.4f (no Mask)\n', t2/t5) ;
    fprintf (' rel: %10.4f ', t2/t5) ;
    fprintf (' speedup: %10.4f\n', t1/t4) ;

    assert (isequal (C, C2)) ;
    assert (isequal (C, C4)) ;
    assert (isequal (C, C5)) ;

    %{

    tic
    C = Mask .* (A'*B) ;
    t1 = toc ;

    % fprintf ('GrB AdotB:\n') ;
    tic
    C2 = GB_mex_AdotB (A,B, Mask) ;
    t2 = toc ;

    % fprintf ('GrB A''*B native:\n') ;
    tic
    C4 = Mask .* GB_mex_AxB (A,B, true) ;
    t4 = toc ;

    % fprintf ('GrB A''*B native:\n') ;
    tic
    C5 = Mask .* GB_mex_AxB (A',B) ;
    t5 = toc ;

    fprintf (...
    'm %2d builtin: %10.4f AdotB : %10.4f   GB,auto:: %10.4f outer %10.4f', ...
    m, t1, t2, t4, t5) ;
    fprintf (' speedup: %10.4f (with Mask)\n', t1/t4) ;

    assert (isequal (C, C2)) ;
    assert (isequal (C, C4)) ;
    %}

    loglog (relwork, reltime, 'o') ;
    drawnow

end
end

fprintf ('\ntest52: all tests passed\n') ;
