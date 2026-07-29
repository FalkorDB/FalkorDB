function C = cat (dim, varargin)
%CAT Concatenate arrays.
% C = cat (dim, A, B) concatenates the two matrices A and B along the dimension
% dim, which must be 1 or 2.  Multidimensional GhB matrices are not supported.
% C = cat (2,A,B) is the same as C = [A,B], and C = cat (1,A,B) is the same as
% C = [A;B].
%
% C = cat (dim, A1, A2, A3 ...) is the same as [A1,A2,A3,...] if dim is 2, and
% [A1;A2;A3;...] if dim is 1.
%
% If A and B are GhB matrices and S = {A B} is a cell array, then
% C = cat (dim, S) does not trigger the GhB/cat method, but uses the built-in
% method instead.  Use GhB.cell2mat instead.
%
% If the matrices have different types, the type is determined according to the
% rules in 'help GrB.optype'.
%
% Example:
%
%   A = GhB (magic (3))
%   B = GhB (pascal (3))
%   C1 = [A ; B]
%   C2 = cat (1, A, B)
%   assert (isequal (C1, C2)) ;
%   C1 = [A B]
%   C2 = cat (2, A, B)
%   assert (isequal (C1, C2)) ;
%
% See also GhB/horzcat, GhB/vertcat, GhB.cell2mat, GhB/mat2cell,
% GhB/num2cell.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_cat (1, dim, varargin {:}) ;

