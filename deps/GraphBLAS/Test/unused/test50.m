function test50
%TEST50 test AxB numeric and symbolic

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('\n----------------------------- GB_mex_AxB\n') ;

% Prob = ssget (2662) ;
Prob = ssget (936)
A = Prob.A ;
A(1,2) = 443 ;

for trial = 1:2

    if (trial == 1)
        B = A ;
    else
        % make it rectangular
        [m k] = size (A) ;
        A = A (:,1:k-1) ;
        [m k] = size (A) ;
        nz = nnz (A) ;
        n = m+1 ;
        B = sprand (k, n, nz / (k*n)) ;
    end

    fprintf ('\n---------------builtin: C=A*B\n') ;
    tic ;
    C = (A*B) ;
    toc
    C0 = cast (C, 'logical') ;
    cnorm = norm (C, 1) ;
    fprintf ('nnz(C) %d density %g mincol %d maxcol %d norm %g\n', ...
        nnz (C0), nnz (C0) / prod (size (C0)), ...
        full (min (sum (spones (C0)))), ...
        full (max (sum (spones (C0)))), cnorm) ;
    % figure (1)
    % subplot (1,3,1) ; spy (A) ;
    % subplot (1,3,2) ; spy (B) ;
    % subplot (1,3,3) ; spy (C) ;

    fprintf ('numerical matrix multiply (GraphBLAS):\n') ;
    tic
    S = GB_mex_AxB (A, B) ;
    toc
    assert (GB_spok (S*1) == 1) ;
    err = norm (C-S,1) / cnorm ;
    fprintf ('err %g\n', err) ;
    assert (isequal (C, S)) ;

    fprintf ('\n---------------builtin: C=(A*B)''\n') ;
    tic ;
    C = (A*B)' ;
    toc
    C0 = cast (C, 'logical') ;

end

fprintf ('\ntest50: all tests passed\n') ;

