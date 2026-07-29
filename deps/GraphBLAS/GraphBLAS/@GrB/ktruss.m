function C = ktruss (varargin)
%GRB.KTRUSS find the k-truss of a graph.
%
% See 'help GhB.ktruss' for details.
% This method is identical, except that it returns GrB matrices.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = GrB (GhB.ktruss (varargin {:})) ;

