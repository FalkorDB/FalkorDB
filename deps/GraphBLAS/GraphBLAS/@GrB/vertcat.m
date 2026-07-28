function C = vertcat (varargin)
%VERTCAT vertical concatenation.
% [A ; B] is the vertical concatenation of A and B.  Multiple matrices may be
% concatenated, as [A ; B ; C ; ...].  If the matrices have different types,
% the type is determined according to the rules in GrB.optype.
%
% See also GrB/horzcat, GrB/cat, GrB.cell2mat, GrB/mat2cell, GrB/num2cell.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_cat (0, 1, varargin {:}) ;

