function C = vertcat (varargin)
%VERTCAT vertical concatenation.
% [A ; B] is the vertical concatenation of A and B.  Multiple matrices may be
% concatenated, as [A ; B ; C ; ...].  If the matrices have different types,
% the type is determined according to the rules in GhB.optype.
%
% See also GhB/horzcat, GhB/cat, GhB.cell2mat, GhB/mat2cell, GhB/num2cell.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_cat (1, 1, varargin {:}) ;

