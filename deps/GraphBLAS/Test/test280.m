function test280(quick)
%TEST280 subassign method 26

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin < 1)
    quick = 0 ;
end

rng ('default') ;

% quick: 0
tic
load west0479
GB_mex_grow (west0479) ;
toc

if (quick >= 1)
    tic
    n = 1e5 ;
    nz = 10e6 ;
    A = sprand (n, n, nz/n^2) ;
    GB_mex_grow (A) ;
    toc
end

if (quick >= 2)
    n = 1e6 ;
    nz = 1e5 ;
    tic
    A = sprand (n, n, nz/n^2) ;
    GB_mex_grow (A) ;
    toc
end

if (quick >= 2)
    n = 2e6 ;
    nz = 10e6 ;
    tic
    A = sprand (n, n, nz/n^2) ;
    GB_mex_grow (A) ;
    toc
end

fprintf ('test280 all tests passed.\n') ;

