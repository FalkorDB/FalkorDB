function DiGraph = digraph (G, option)
%DIGRAPH convert a GraphBLAS matrix into a directed DiGraph.
% DiGraph = digraph (G) converts a GraphBLAS matrix G into a directed
% DiGraph.  G must be square.  If G is logical, then no weights are added
% to the DiGraph.  If G is single or double, these become the weights of
% the DiGraph.  If G is integer, the DiGraph is constructed with weights
% of type double.
%
% DiGraph = digraph (G, 'omitselfloops') ignores the diagonal of G, and
% the resulting DiGraph has no self-edges.  The default is that
% self-edges are created from any diagonal entries of G.
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

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

G = G.opaque ;

[m, n, type] = gbsize (G) ;
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
    G = gbselect ('offdiag', G, 0) ;
end

% construct the digraph
switch (type)

    case { 'single' }

        % The digraph(...) function can accept x as single, but not
        % from a sparse matrix.  So extract the tuples of G first.
        [i, j, x] = gbextracttuples (G) ;
        DiGraph = digraph (i, j, x, n) ;

    case { 'logical' }

        % The digraph(...) function allows for logical
        % adjacency matrices (no edge weights are created).
        DiGraph = digraph (gbbuiltin (G, 'logical')) ;

    otherwise

        % typecast to double
        DiGraph = digraph (gbbuiltin (G, 'double')) ;
end

