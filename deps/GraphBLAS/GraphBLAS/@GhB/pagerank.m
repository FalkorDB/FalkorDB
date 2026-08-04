function [r, stats] = pagerank (A, opts)
%GHB.PAGERANK PageRank of a graph.
% r = GhB.pagerank (A) computes the PageRank of a graph with adjacency matrix
% A.  r = GhB.pagerank (A, opts) allows for non-default options to be selected.
% For compatibility with the built-in methods, defaults are identical to the
% built-in pagerank method in @graph/centrality and @digraph/centrality:
%
%   opts.tol = 1e-4         stopping criterion
%   opts.maxit = 100        maximum # of iterations to take
%   opts.damp = 0.85        dampening factor
%   opts.weighted = false   true: use edgeweights of A; false: use spones(A)
%   opts.type = 'double'    compute in 'single' or 'double' precision
%   opts.pnorm = inf        1, 2, or inf: selects the norm to use to check for
%       convergence.  The MATLAB centrality (..,'pagerank') uses inf, but the
%       GAP benchmark, LAGraph, and most other pagerank methods use the 1-norm.
%
% An optional 2nd output argument provides statistics:
%   stats.tinit     initialization time
%   stats.trank     pagerank time
%   stats.iter      # of iterations taken
%
% See also graph/centrality, digraph/centrality.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% This method is typically faster if A is held by column.  It is not the GAP
% algorithm, but a proper pagerank method that correctly handles the sinks.
% The GAP benchmark method ignores sinks, and its pagerank does not ensure
% sum(r)=1.  Handling the sinks takes more work, so this method should not be
% benchmarked against the GAP pagerank.  See the two LAGraph pagerank methods;
% one is the GAP method; the other is this method.

%-------------------------------------------------------------------------------
% initializations
%-------------------------------------------------------------------------------

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
if (~isfield (opts, 'pnorm'))
    opts.pnorm = inf ;
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
pnorm = opts.pnorm ;

[m, n] = size (A) ;
if (m ~= n)
    error ('GrB:error', 'A must be square') ;
end

% select the semiring
if (weighted)
    % use the weighted edges of A
    semiring = ['+.*.' type] ;
else
    % use just the pattern of A
    semiring = ['+.2nd.' type] ;
end

% select the accum operator, according to the type
accum = ['+.' type] ;

if (weighted)
    % d (i) = sum (A (i,:))
    d = GhB (n, 1, type, 'bitmap/full') ;
    GhB.vreduce (d, A, '+') ;
else
    % d (i) = outdegree of node i
    d = GhB (GhB.entries (A, 'row', 'degree'), type, 'bitmap/full') ;
end

% look for sinks and revise d
sinks = find (d == 0) ;
any_sinks = ~isempty (sinks) ;
if (any_sinks)
    % see d (sinks) = 1, to avoid divide-by-zero
    GhB.subassign (d, { sinks }, 1) ;       % d (sinks) = 1
end

%-------------------------------------------------------------------------------
% compute the pagerank
%-------------------------------------------------------------------------------

stats.tinit = toc (tstart) ;
tstart = tic ;

% teleport factor, assuming no sinks
tfactor = cast ((1 - damp) / n, type) ;

% sink factor
dn = cast (damp / n, type) ;

% use A' in GhB.mxm
desc.in0 = 'transpose' ;

% initial PageRank r: all nodes have rank 1/n
r = GhB (n, 1, type) ;              % r = sparse (n,1,type)
GhB.assign (r, { }, 1/n) ;          % r (:) = 1/n

% prescale d with damp so it doesn't have to be done in each iteration
GhB.apply2 (d, d, '/', damp) ;      % d = d / damp ;

% compute the PageRank
for iter = 1:maxit
    prior = r ;                                 % prior = r ; a handle alias
    teleport = tfactor ;
    if (any_sinks)
        % add the teleport factor from all the sinks
        % teleport += dn * sum (r (sinks))) ;
        teleport = teleport + dn * sum (GhB.extract (r, { sinks })) ;
    end
    r = GhB.expand (teleport, prior) ;          % r (1:n) = teleport
    t = GhB.emult (prior, '/', d) ;             % t = prior ./ d
    GhB.mxm (r, accum, A, semiring, t, desc) ;  % r += A' * t
    e = GhB.normdiff (r, prior, pnorm) ;        % e = norm (r-prior, pnorm)
    if (e < tol)
        % convergence has been reached
        stats.trank = toc (tstart) ;
        stats.iter = iter ;
        return ;
    end
end

warning ('GrB:pagerank', 'pagerank failed to converge') ;

