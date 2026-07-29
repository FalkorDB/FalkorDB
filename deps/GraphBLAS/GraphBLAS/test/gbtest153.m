function gbtest153
%GBTEST153 test GhB.apply2 (not inplace but with pending work)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[filepath, name, ext] = fileparts (mfilename ('fullpath')) ;
load (fullfile (filepath, './matrix/west0479_correct.mat')) ;

A = Problem.A ;
[i,j,x] = find (A) ;
n = size (A, 1) ;
nz = length (x) ;

G = GhB (n, n) ;
for k = 1:nz
    G (i (k), j (k)) = x (k) ;
end

% pass in a matrix G with pending work to GhB.apply2
% as the first input matrix.

% fprintf ('apply2:\n') ;
% G = 2*G
G = GhB.apply2 (G, 2, '*', G) ;

assert (isequal (G, 2*A)) ;

fprintf ('gbtest153: all tests passed\n') ;

