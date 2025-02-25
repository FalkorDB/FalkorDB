%TEST_SAXPY3 test integer overflow in GB_AxB_saxpy3_cumsum.c
%
% This test creates successively larger matrices, eventually triggering
% integer overflow when Cp is 32-bit and the matrix has more the 2^32 entries.
% It does not need to set the global hack flag to trigger the condition
% artificially.  However, this test is costly and can only be done on a
% large system, so it is not part of the testall.m script.  See test296.m.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;
clear all
clear mex
n = 10e6 ;
nz = 1 ;
range = int8 ([0 128]) ;
GrB.threads (32) ;
% GrB.burble (1) ;
e = 1 ;

while (e < 4.2e9)

    nz = 1.25*nz ;
    d = nz / n^2 ;
    tic ;
    C = GrB.random (n, n, d, 'range', range) ;
    t = toc ;
    e = GrB.entries (C) ;
    fprintf ('\ncreate: %g sec, nvals %g million\n', t, e / 1e6) ;
    tic ;
    C = C^2 ;
    t = toc ;
    e = GrB.entries (C) ;
    fprintf ('mxm: %g sec, nvals: %g million\n', t, e/1e6) ;
%   disp (C, 1) ;
    clear C

end
