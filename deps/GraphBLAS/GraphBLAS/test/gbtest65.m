function gbtest65 (ghb)
%GBTEST65 test [GrB,GhB].mis

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

[filepath, name, ext] = fileparts (mfilename ('fullpath')) ;
load (fullfile (filepath, './matrix/west0479_correct.mat')) ;

A = gtb_offdiag (ghb, spones (Problem.A)) ;
A = A+A' ;

n = size (A, 1) ;

for byrow = 0:1

    if (byrow)
        fprintf ('input matrix held by col:\n') ;
    else
        fprintf ('input matrix held by row:\n') ;
        A = gtb (ghb, A, 'by row') ;
    end
    maxisize = 0 ;
    rng ('default') ;

    for trial = 1:100

        if (mod (trial, 4) == 1)
            iset  = gtb_mis (ghb, A, 'check') ;
        else
            iset  = gtb_mis (ghb, A) ;
        end

        % assert that iset is an independent set
        p = find (iset) ;
        assert (nnz (A (p,p)) == 0) ;
        isize = length (p) ;

        if (isize > maxisize)
            fprintf ('trial %3d: iset size: %d\n', trial, isize) ;
            maxisize = isize ;
        end

        % assert that iset is maximal
        q = find (~iset) ;
        d = gtb_entries (ghb, A (p, q), 'col', 'degree') ; %#ok<FNDSB>
        assert (all (d > 0)) ;
    end

    fprintf ('max independent set found: %d of %d nodes\n', maxisize, n) ;
end

fprintf ('gbtest65 (%d): all tests passed\n', ghb) ;

