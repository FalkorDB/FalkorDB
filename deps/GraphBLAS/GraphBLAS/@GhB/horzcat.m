function C = horzcat (varargin)
%HORZCAT horizontal concatenation.
% [A B] or [A,B] is the horizontal concatenation of A and B.  Multiple matrices
% may be concatenated, as [A, B, C, ...].  If the matrices have different
% types, the type is determined according to the rules in GrB.optype.
%
% See also GhB/vertcat, GhB/cat, GhB.cell2mat, GhB/mat2cell, GhB/num2cell.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_cat (1, 0, varargin {:}) ;

