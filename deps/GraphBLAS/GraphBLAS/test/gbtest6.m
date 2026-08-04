function gbtest6 (ghb)
%GBTEST6 test GrB.mxm and GhB.mxm

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

A = sparse (rand (2)) ;
B = sparse (rand (2)) ;

C = A*B ;

G = gtb_mxm (ghb, '+.*', A, B) ;
err = norm (C-G, 1) ;
assert (err < 1e-12) ;

d.kind = 'sparse' ;
d.in0 = 'transpose' ;
G = gtb_mxm (ghb, '+.*', A, B, d) ;
C = A'*B ;

err = norm (C-G, 1) ;
assert (err < 1e-12) ;

d.kind = 'GrB' ;
G = gtb_mxm (ghb, '+.*', A, B, d) ;
err = norm (C-G, 1) ;
assert (err < 1e-12) ;
clear d

E = sparse (rand (2)) ;
C = E + A*B ;
G = gtb_mxm (ghb, E, '+', '+.*', A, B) ;
err = norm (C-G, 1) ;
assert (err < 1e-12) ;

M = false (2,2) ;
Cin = rand (2) ;
M (1,1) = 1 ;
G = gtb_mxm (ghb, Cin, M, '+', '+.*', A, B) ;
T = Cin + A*B ;
C = Cin ;
C (M) = T (M) ;
err = norm (C-G, 1) ;
assert (err < 1e-12)

n = 10 ;
A = sprand (n, n, 0.1) ;
B = rand (n) ;
G = gtb_mxm (ghb, '+.*', A, B) ;
E = gtb (ghb, A) * B ;
C = A*B ;
err = norm (C-G, 1) ;
err = norm (E-G, 1) ;
assert (err < 1e-12) ;

% G is exported as a MATLAB/Octave sparse matrix, but as double-complex instead
% of single-complex, since MATLAB/Octave do not yet have sparse single complex
% matrices (at least earlier versions of those packages).
clear d
d.kind = 'builtin' ;
A = A + 1i * sprand (n, n, 0.1) ;
A = gtb (ghb, A, 'single complex') ;
G = gtb_mxm (ghb, '+.*', A, A, d) ;
B = complex (A) ;
C = B*B ;
err = norm (C-G, 1) ;
assert (err < 1e-6) ;
assert (isequal (gtb_type (ghb, G), 'double complex')) ;
[f,s] = gtb_format (ghb, G) ;
assert (isequal (s, 'sparse')) ;

% full matrices can be exported as single complex MATLAB/Octave matrices
d.kind = 'full' ;
G = gtb_mxm (ghb, '+.*', A, A, d) ;
assert (isequal (gtb_type (ghb, G), 'single complex')) ;
[f,s] = gtb_format (ghb, G) ;
assert (isequal (s, 'full')) ;

fprintf ('gbtest6 (%d): all tests passed\n', ghb) ;

