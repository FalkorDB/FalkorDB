function [r, stats] = pagerank (A, opts)
%GRB.PAGERANK PageRank of a graph.
% r = GrB.pagerank (A) computes the PageRank of a graph with adjacency
% matrix A.  r = GrB.pagerank (A, options) allows for non-default options
% to be selected.  For compatibility with the built-in methods, defaults
% are identical to the built-in pagerank method in @graph/centrality and
% @digraph/centrality:
%
%   opts.tol = 1e-4         stopping criterion
%   opts.maxit = 100        maximum # of iterations to take
%   opts.damp = 0.85        dampening factor
%   opts.weighted = false   true: use edgeweights of A; false: use spones(A)
%   opts.type = 'double'    compute in 'single' or 'double' precision
%
% A can be a GraphBLAS or built-in matrix.  A can have any format ('by row'
% or 'by col'), but GrB.pagerank is faster if A is 'by col'.
%
% An optional 2nd output argument provides statistics:
%   stats.tinit     initialization time
%   stats.trank     pagerank time
%   stats.iter      # of iterations taken
%
% See also graph/centrality.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% NOTE: this is a high-level algorithm that uses GrB objects.

%-------------------------------------------------------------------------
% initializations
%-------------------------------------------------------------------------

tstart = tic ;

% check inputs and set defaults
if (nargin < 2)
    opts = struct ;
end
if (~isfield (opts, 'tol'))
    opts.tol = 1e-4 ;
end
if (~isfield (opts, 'maxit'))
    opts.maxit = 100 ;
end
if (~isfield (opts, 'damp'))
    opts.damp = 0.85 ;
end
if (~isfield (opts, 'weighted'))
    opts.weighted = false ;
end
if (~isfield (opts, 'type'))
    opts.type = 'double' ;
end

if (~(isequal (opts.type, 'single') || isequal (opts.type, 'double')))
    error ('GrB:error', 'opts.type must be ''single'' or ''double''') ;
end

% get options
tol = opts.tol ;
maxit = opts.maxit ;
damp = opts.damp ;
damp = max (damp, 0) ;
damp = min (damp, 1) ;
type = opts.type ;
weighted = opts.weighted ;

[m, n] = size (A) ;
if (m ~= n)
    error ('GrB:error', 'A must be square') ;
end

% select the semiring and determine if A is native
if (weighted)
    % use the weighted edges of G
    % native, if A is already of the right type, and stored by column
    native = (GrB.isbycol (A) & isequal (GrB.type (A), type)) ;
    semiring = ['+.*.' type] ;
else
    % use just the pattern of G, so A can be of any type.
    % native, if A is already stored by column
    native = GrB.isbycol (A) ;
    semiring = ['+.2nd.' type] ;
end

% construct the matrix G, or use A as-is
if (native)
    G = A ;
else
    G = GrB (A, type, 'by col') ;
end

% select the accum operator, according to the type
accum = ['+.' type] ;

% d (i) = outdegree of node i, or 1 if i is a sink
d = GrB (GrB.entries (A, 'row', 'degree'), type) ;
sinks = find (d == 0) ;
any_sinks = ~isempty (sinks) ;
if (any_sinks)
    % d (sinks) = 1, to avoid divide-by-zero
    d = GrB.subassign (d, { sinks }, 1) ;
end

%-------------------------------------------------------------------------
% compute the pagerank
%-------------------------------------------------------------------------

stats.tinit = toc (tstart) ;
tstart = tic ;

% teleport factor
tfactor = cast ((1 - damp) / n, type) ;

% sink factor
dn = cast (damp / n, type) ;

% use G' in GrB.mxm
desc.in0 = 'transpose' ;

% initial PageRank: all nodes have rank 1/n
r = GrB (ones (n, 1, type) / n) ;

% prescale d with damp so it doesn't have to be done in each iteration
d = d / damp ;

% compute the PageRank
for iter = 1:maxit
    prior = r ;
    teleport = tfactor ;
    if (any_sinks)
        % add the teleport factor from all the sinks
        % teleport = teleport + dn * sum (r (sinks))) ;
        teleport = teleport + dn * sum (GrB.extract (r, { sinks })) ;
    end
    % r (1:n) = teleport
    r = GrB.expand (teleport, r) ;
    % t = prior ./ d
    t = GrB.emult (prior, '/', d) ;
    % r = r + G' * t
    r = GrB.mxm (r, accum, G, semiring, t, desc) ;
    % e = norm (r-prior, inf)
    e = GrB.normdiff (r, prior, inf) ;
    if (e < tol)
        % convergence has been reached
        stats.trank = toc (tstart) ;
        stats.iter = iter ;
        return ;
    end
end

warning ('GrB:pagerank', 'pagerank failed to converge') ;

