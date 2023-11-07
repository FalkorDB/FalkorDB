function [v, parent] = bfs (A, s, varargin)
%GRB.BFS breadth-first search of a graph, using its adjacency matrix.
% v = GrB.bfs (A, s) performs the breadth-first search of the directed
% graph represented by the square adjacency matrix A.  The breadth-first
% search starts at node s.  The output v is a sparse vector of size
% n-by-1, with the level of each node, where v(s)=1, and v(i)=k if the
% path with the fewest edges from from s to i has k-1 edges.  If i is not
% reachable from s, then v(i) is implicitly zero and does not appear in
% the pattern of v.
%
% [v, parent] = GrB.bfs (A, s) also computes the parent vector,
% representing the breadth-first search tree.  parent(s)=s denotes the
% root of the tree, and parent(c)=p if node p is the parent of c in the
% tree.  The parent vector is sparse, and parent (i) is not present if i
% is not found in the breadth-first search.
%
% Optional string arguments can be provided, after A and s:
%
%   'undirected' or 'symmetric':  A is assumed to be symmetric, and
%       represents an undirected graph.  Results are undefined if A is
%       unsymmetric, and 'check' is not specified.
%
%   'directed' or 'unsymmetric':  A is assumed to be unsymmetric, and
%       presents a directed graph.  This is the default.
%
%   'check': with the 'undirected' or 'symmetric' option, A is checked to
%       ensure that it is symmetric.  The default is not to check.
%
% For best performance, if A represents a directed graph, it should be a
% GraphBLAS matrix stored by row on input.  That is, GrB.format (A)
% should report 'by row'.  (If A represents a directed graph but is
% stored 'by col' on input, it is first converted to 'by row', which is
% costly).  If A is an undirected graph, then it can be stored in either
% format ('by row' or 'by col').
%
% A must be square.  Only the pattern, spones (A), is considered; the
% values of its entries (the edge weights of the graph) are ignored.
%
% Example:
%
%   A = bucky ;
%   s = 1 ;
%   [v pi] = GrB.bfs (A, s)
%   figure (1) ;
%   subplot (1,2,1) ; plot (graph (A)) ;
%   pi2 = full (double (pi)) ;
%   pi2 (s) = 0 ;
%   subplot (1,2,2) ; treeplot (pi2) ; title ('BFS tree') ;
%   n = size (A,1) ;
%   for level = 1:n
%       level
%       inlevel = find (v == level)
%       parents = full (double (pi (inlevel)))
%       if (isempty (inlevel))
%           break ;
%       end
%   end
%
% See also graph/bfsearch, graph/shortestpathtree, treeplot.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% NOTE: this is a high-level algorithm that uses GrB objects.

%-------------------------------------------------------------------------
% initializations
%-------------------------------------------------------------------------

[m, n] = size (A) ;
if (m ~= n)
    error ('GrB:error', 'A must be square') ;
end

% get the string options
kind = 'directed' ;
check = false ;
for k = 1:nargin-2
    arg = lower (varargin {k}) ;
    switch arg
        case { 'undirected', 'symmetric' }
            kind = 'undirected' ;
        case { 'directed', 'unsymmetric' }
            kind = 'directed' ;
        case { 'check' }
            check = true ;
        otherwise
            error ('GrB:error', 'unknown option') ;
    end
end

% set the descriptors
desc_rc.out  = 'replace' ;
desc_rc.mask = 'complement' ;
desc_s.mask = 'structural' ;

% determine the method to use, and convert A if necessary
if (isequal (kind, 'undirected'))
    if (check && ~issymmetric (A))
        error ('GrB:error', 'A must be symmetric') ;
    end
    if (GrB.isbycol (A))
        % A is stored by column but undirected, so use q*A' instead of q*A
        desc_rc.in1 = 'transpose' ;
    end
else
    if (GrB.isbycol (A))
        % this can be costly
        A = GrB (A, 'by row') ;
    end
end

% determine the integer type to use
int_type = 'int64' ;
if (n < intmax ('int32'))
    int_type = 'int32' ;
end

% initialize v as a full integer vector
v = full (GrB (1, n, int_type)) ;

%-------------------------------------------------------------------------
% do the BFS
%-------------------------------------------------------------------------

if (nargout == 1)

    %---------------------------------------------------------------------
    % just compute the level of each node
    %---------------------------------------------------------------------

    q = GrB (1, n, 'logical') ;                  % q = sparse (1,n)
    q = GrB.subassign (q, { s }, true) ;         % q (s) = 1
    for level = 1:n
        % assign the current level: v<q> = level
        v = GrB.subassign (v, q, level, desc_s) ;
        % quit if q is empty
        if (~any (q)), break, end
        % move to the next level:  q<~v,replace> = q*A
        q = GrB.mxm (q, v, 'any.pair.logical', q, A, desc_rc) ;
    end

else

    %---------------------------------------------------------------------
    % compute both the level and the parent
    %---------------------------------------------------------------------

    parent = full (GrB (1, n, int_type)) ;       % parent = zeros (1,n)
    parent = GrB.subassign (parent, { s }, s) ;  % parent (s) = s
    q = GrB (1, n, int_type) ;                   % q = sparse (1,n)
    q = GrB.subassign (q, { s }, s) ;            % q (s) = s
    id = GrB (1:n, int_type, 'by row') ;         % id = 1:n
    semiring = ['any.1st.' int_type] ;           % any.1st.integer semiring
    for level = 1:n
        % assign the current level: v<q> = level
        v = GrB.subassign (v, q, level, desc_s) ;
        % quit if q is empty
        if (~any (q)), break, end
        % move to the next level:  q<~v,replace> = q*A,
        % using the any-first-integer semiring (int32 or int64)
        q = GrB.mxm (q, v, semiring, q, A, desc_rc) ;
        % assign parents: parent<q> = q
        parent = GrB.assign (parent, q, q, desc_s) ;
        % q(i) = i for all entries in q, using q<q>=1:n
        q = GrB.assign (q, q, id, desc_s) ;
    end
    % remove zeros from parent
    parent = GrB.prune (parent) ;

end

% remove zeros from v
v = GrB.prune (v) ;

