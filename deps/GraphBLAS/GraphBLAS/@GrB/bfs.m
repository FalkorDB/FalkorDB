function [varargout] = bfs (varargin)
%GRB.BFS breadth-first search of a graph, using its adjacency matrix.
%
% See 'help GhB.bfs' for details.
% This method is identical, except that it returns GrB matrices.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[varargout{1:nargout}] = GhB.bfs (varargin {:}) ;
for k = 1:nargout
    varargout {k} = GrB (varargout {k}) ;
end

