function gbtest154
%GBTEST154 test GrB.bytes

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[filepath, name, ext] = fileparts (mfilename ('fullpath')) ;
load (fullfile (filepath, './matrix/west0479_correct.mat')) ;

A = Problem.A ;

GrB.print (A) ;

[m n] = size (A) ;
e = nnz (A) ;

% using 64-bit integers
mem = (n+1)*8 + 16*e ;
s = GrB.bytes (A) ;

fprintf ('MATLAB bytes: %g %g diff: %g\n', mem, s, s-mem) ;
assert (s >= mem && s < mem + 1000) ;

% using 32-bit integers
G = GrB (A) ;
mem = (n+1)*4 + 12*e ;
s = GrB.bytes (G) ;

fprintf ('GrB bytes: %g %g diff: %g\n', mem, s, s-mem) ;
assert (s >= mem && s < mem + 1000) ;

H = GhB (A) ;
s = GhB.bytes (H) ;
fprintf ('GhB bytes: %g %g diff: %g\n', mem, s, s-mem) ;
assert (s >= mem && s < mem + 1000) ;

whos

fprintf ('gbtest154: all tests passed\n') ;

