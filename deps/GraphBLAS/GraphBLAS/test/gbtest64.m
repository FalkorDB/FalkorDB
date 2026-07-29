function gbtest64 (ghb)
%GBTEST64 test [GrB,GhB].pagerank

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

[filepath, name, ext] = fileparts (mfilename ('fullpath')) ;
load (fullfile (filepath, './matrix/west0479_correct.mat')) ;

W = abs (Problem.A) ;
W (1,:) = 0 ;

A = digraph (W) ;
G = gtb (ghb, W) ;
R = gtb (ghb, W, 'by row') ;

r1 = centrality (A, 'pagerank') ;
r2 = gtb_pagerank (ghb, G) ;
assert (norm (r1-r2) < 1e-12) ;

r1 = centrality (A, 'pagerank') ;
r2 = gtb_pagerank (ghb, R) ;
assert (norm (r1-r2) < 1e-12) ;

r1 = centrality (A, 'pagerank', 'Tolerance', 1e-8) ;
r2 = gtb_pagerank (ghb, G, struct ('tol', 1e-8)) ;
assert (norm (r1-r2) < 1e-12) ;

lastwarn ('') ;
warning ('off', 'MATLAB:graphfun:centrality:PageRankNoConv') ;
warning ('off', 'GrB:pagerank') ;

r1 = centrality (A, 'pagerank', 'MaxIterations', 2) ;
[msg, id] = lastwarn ; %#ok<*ASGLU>

r2 = gtb_pagerank (ghb, G, struct ('maxit', 2)) ;
[msg, id] = lastwarn ;
assert (isequal (id, 'GrB:pagerank')) ;
assert (norm (r1-r2) < 1e-12) ;

lastwarn ('') ;

r1 = centrality (A, 'pagerank', 'FollowProbability', 0.5) ;
r2 = gtb_pagerank (ghb, G, struct ('damp', 0.5)) ;
assert (norm (r1-r2) < 1e-12) ;

r1 = gtb_pagerank (ghb, G, struct ('weighted', true)) ;
r2 = gtb_pagerank (ghb, R, struct ('weighted', true)) ;
assert (norm (r1-r2) < 1e-12) ;

fprintf ('gbtest64 (%d): all tests passed\n', ghb) ;

