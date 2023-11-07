function gbtest64
%GBTEST64 test GrB.pagerank

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% load west0479 ; %#ok<*LOAD>
load west0479_correct ; %#ok<*LOAD>
west0479 = Problem.A ;
W = abs (west0479) ;
W (1,:) = 0 ;

A = digraph (W) ;
G = GrB (W) ;
R = GrB (W, 'by row') ;

r1 = centrality (A, 'pagerank') ;
r2 = GrB.pagerank (G) ;
assert (norm (r1-r2) < 1e-12) ;

r1 = centrality (A, 'pagerank') ;
r2 = GrB.pagerank (R) ;
assert (norm (r1-r2) < 1e-12) ;

r1 = centrality (A, 'pagerank', 'Tolerance', 1e-8) ;
r2 = GrB.pagerank (G, struct ('tol', 1e-8)) ;
assert (norm (r1-r2) < 1e-12) ;

lastwarn ('') ;
warning ('off', 'MATLAB:graphfun:centrality:PageRankNoConv') ;
warning ('off', 'GrB:pagerank') ;

r1 = centrality (A, 'pagerank', 'MaxIterations', 2) ;
[msg, id] = lastwarn ; %#ok<*ASGLU>

r2 = GrB.pagerank (G, struct ('maxit', 2)) ;
[msg, id] = lastwarn ;
assert (isequal (id, 'GrB:pagerank')) ;
assert (norm (r1-r2) < 1e-12) ;

lastwarn ('') ;

r1 = centrality (A, 'pagerank', 'FollowProbability', 0.5) ;
r2 = GrB.pagerank (G, struct ('damp', 0.5)) ;
assert (norm (r1-r2) < 1e-12) ;

r1 = GrB.pagerank (G, struct ('weighted', true)) ;
r2 = GrB.pagerank (R, struct ('weighted', true)) ;
assert (norm (r1-r2) < 1e-12) ;

fprintf ('gbtest64: all tests passed\n') ;

