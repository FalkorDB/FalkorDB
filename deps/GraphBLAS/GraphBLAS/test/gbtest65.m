function gbtest65
%GBTEST65 test GrB.mis

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

load west0479_correct.txt
west0479 = spconvert (west0479_correct) ;

A = GrB.offdiag (spones (west0479)) ;
A = A+A' ;

maxisize = 0 ;
n = size (A, 1) ;

for trial = 1:100

    if (mod (trial, 4) == 1)
        iset  = GrB.mis (A, 'check') ;
    else
        iset  = GrB.mis (A) ;
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
    d = GrB.entries (A (p, q), 'col', 'degree') ; %#ok<FNDSB>
    assert (all (d > 0)) ;
end

fprintf ('max independent set found: %d of %d nodes\n', maxisize, n) ;

fprintf ('gbtest65: all tests passed\n') ;

