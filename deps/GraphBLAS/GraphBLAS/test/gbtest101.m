function gbtest101 (ghb)
%GBTEST101 test loading of v3 and v10.3.1 GraphBLAS objects

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

[filepath, name, ext] = fileparts (mfilename ('fullpath')) ;
load ([filepath '/gbtest101_matfiles/gbtestv3.mat']) ; %#ok<LOAD>
whos

fprintf ('================== v3 sparse:\n') ;
G
fprintf ('================== latest sparse:\n') ;
G2 = gtb (ghb, G, 'sparse') ;
assert (isequal (G, A)) ;
assert (isequal (G2, A)) ;

[m1, n1] = size (G) ;
[m2, n2] = size (A) ;
assert (m1 == m2) ;
assert (n1 == n2) ;

t1 = gtb_type (ghb, G) ;
t2 = gtb_type (ghb, A) ;
assert (isequal (t1, t2)) ;

[s1, f1] = gtb_format (ghb, G) ;
[s2, f2, iso] = gtb_format (ghb, G2) ;
assert (isequal (s1, s2)) ;
assert (isequal (f1, f2)) ;
iso

H2 = gtb (ghb, H, 'hyper') ;
fprintf ('================== v3 hypersparse:\n') ;
H
fprintf ('================== latest hypersparse:\n') ;
H2

H3 = gtb (ghb, n,n) ;
H3 (1:4, 1:4) = magic (4) ;
assert (isequal (H2, H)) ;
assert (isequal (H3, H)) ;

[s1, f1] = gtb_format (ghb, H) ;
[s2, f2] = gtb_format (ghb, H2) ;
assert (isequal (s1, s2)) ;
assert (isequal (f1, f2)) ;

t1 = gtb_type (ghb, H2) ;
t2 = gtb_type (ghb, H) ;
assert (isequal (t1, t2)) ;

R2 = gtb (ghb, R) ;
assert (isequal (R2, R)) ;
assert (isequal (R2, A')) ;

X2 = gtb (ghb, X) ;
assert (isequal (magic (4), X)) ;
assert (isequal (magic (4), X2)) ;

fprintf ('================== v3 dense (held in sparse format):\n') ;
X
fprintf ('================== latest dense:\n') ;
X2

% test GrB/struct:
S = struct (G)
assert (isstruct (S)) ;
assert (isfield (S, 'opaque') || isfield (S, 'GraphBLASv10')) ;

% test [GrB,GhB]/struct:
S = struct (G2)
assert (isstruct (S)) ;
if (isfield (S, 'opaque'))
    assert (isequal (size (S.opaque), [1 8])) ;
    assert (isequal (class (S.opaque), 'uint8')) ;
end

% H was constructed in GraphBLAS v10.3.1 as:
% n = 2^60 ; H = GrB (n,n) ;
% load ./matrix/west0479_correct ;
% H (1:479,1:479) = GrB (Problem.A) ;
% k = 2000 ; H (1:k,1:k) = speye (k)
clear H H2 H3
load ([filepath '/gbtest101_matfiles/gbtestv10_3_1.mat']) ; %#ok<LOAD>

% Now construct H again
% [filepath, name, ext] = fileparts (mfilename ('fullpath')) ;
load (fullfile (filepath, './matrix/west0479_correct.mat')) ;

A = gtb (ghb, Problem.A) ;
n = 2^60 ;
H2 = gtb (ghb, n,n) ;
H2 (1:479, 1:479) = A ;
k = 2000 ;
H2 (1:k, 1:k) = speye (k) ;
assert (isequal (H, H2)) ;
[f,s] = gtb_format (ghb, H) ;
assert (isequal (s, 'hypersparse')) ;

% G was constructed in GraphBLAS v10.3.1 as:
% load ./matrix/west0479_correct ;
% G = GrB (Problem.A, 'bitmap') ;
clear G G2
load ([filepath '/gbtest101_matfiles/gbtestv10_3_1b.mat']) ; %#ok<LOAD>

% Now construct G again
G2 = gtb (ghb, A, 'bitmap') ;
assert (isequal (G, G2)) ;
[f,s] = gtb_format (ghb, G) ;
assert (isequal (s, 'bitmap')) ;

fprintf ('gbtest101 (%d): all tests passed\n', ghb) ;

