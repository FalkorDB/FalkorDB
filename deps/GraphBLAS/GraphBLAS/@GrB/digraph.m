function DiGraph = digraph (G, option)
%DIGRAPH convert a GraphBLAS matrix into a directed DiGraph.
% DiGraph = digraph (G) converts a GraphBLAS matrix G into a directed DiGraph.
% G must be square.  If G is logical, then no weights are added to the DiGraph.
% If G is single or double, these become the weights of the DiGraph.  If G is
% integer, the DiGraph is constructed with weights of type double.
%
% DiGraph = digraph (G, 'omitselfloops') ignores the diagonal of G, and the
% resulting DiGraph has no self-edges.  The default is that self-edges are
% created from any diagonal entries of G.
%
% Example:
%
%   G = GrB (sprand (8, 8, 0.2))
%   DiGraph = digraph (G)
%   h = plot (DiGraph) ;
%   h.NodeFontSize = 20 ;
%   h.ArrowSize = 20 ;
%   h.LineWidth = 2 ;
%   h.EdgeColor = [0 0 1] ;
%   t = title ('random directed graph with 8 nodes') ;
%   t.FontSize = 20 ;
%
% See also graph, digraph, GrB/graph.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (G))
    G = struct (G) ;
end

[m, n, type] = gbmex_size (G) ;
if (m ~= n)
    error ('GrB:error', 'G must be square') ;
end

% get the string options
omitself = false ;
if (nargin > 1)
    if (isequal (lower (option), 'omitselfloops'))
        omitself = true ;
    else
        error ('GrB:error', 'unknown option') ;
    end
end

% apply the options
if (omitself)
    % ignore diagonal entries of G
    G2 = gzb_select (1, 'offdiag', G, 0) ;
else
    % use G as-is
    G2 = G ;
end

% construct the graph
switch (type)
    case { 'single', 'logical' }
        gtype = type ;
    otherwise
        gtype = 'double' ;
end

% construct the digraph
switch (type)

    case { 'single' }

        % The digraph(...) function can accept x as single, but not
        % from a sparse matrix.  So extract the tuples of G first.
        gbmex_wait (G2) ;
        [i, j, x] = gbmex_extracttuples (1, G2) ;
        DiGraph = digraph (i, j, x, n) ;

    case { 'logical' }

        % The digraph(...) function allows for logical
        % adjacency matrices (no edge weights are created).
        DiGraph = digraph (gb_builtin (gzb_cast (G2, 'logical'))) ;

    otherwise

        % typecast to double
        DiGraph = digraph (gb_builtin (gzb_cast (G2, 'double'))) ;
end

