function [varargout] = pagerank (varargin)
%GRB.PAGERANK PageRank of a graph.
%
% See 'help GhB.pagerank' for details.
% This method is identical, except that it returns GrB matrices.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[varargout{1:nargout}] = GhB.pagerank (varargin {:}) ;
varargout {1} = GrB (varargout {1}) ;

