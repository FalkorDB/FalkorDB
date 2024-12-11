function test280(quick)
%TEST280 subassign method 26

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin < 1)
    quick = 1 ;
end

rng ('default') ;

tic
load west0479
GB_mex_grow (west0479) ;
toc

tic
n = 10000 ;
nz = 10e6 ;
A = sprand (n, n, nz/n^2) ;
GB_mex_grow (A) ;
toc

tic
if (quick)
    n = 1e6 ;
    nz = 1e5 ;
else
    n = 2e6 ;
    nz = 10e6 ;
end
A = sprand (n, n, nz/n^2) ;
GB_mex_grow (A) ;
toc

fprintf ('test280 all tests passed.\n') ;

