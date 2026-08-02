function iset = mis (varargin)
%GRB.MIS variant of Luby's maximal independent set algorithm.
%
% See 'help GhB.mis' for details.
% This method is identical, except that it returns GrB matrices.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

iset = GrB (GhB.mis (varargin {:})) ;

