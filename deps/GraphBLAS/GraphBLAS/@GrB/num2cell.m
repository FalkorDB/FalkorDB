function C = num2cell (A, dim)
%NUM2CELL Convert matrix into cell array.
%
% C = num2cell (A) converts a GrB matrix A into a cell array C by placing each
%       entry of A in a separate cell, C{i,j} = A(i,j).
%
% C = num2cell (A, 1) creates a 1-by-n cell array, where A is m-by-n, and C{j}
%       is the jth column of A; that is, C{j} = A(:,j).
%
% C = num2cell (A, 2) creates an m-by-1 cell array where C{i} is the ith row of
%       A; that is, C{i} = A (i,:).
%
% C = num2cell (A, [1 2]) constructs a 1-by-1 cell array C with C{1}=A.
%
% C = num2cell (A, [2 1]) constructs a 1-by-1 cell array C with C{1}=A.',
%       the array transpose of A.
%
% See also GrB/horzcat, GrB/vertcat, GrB/cat, GrB.cell2mat, GrB/mat2cell.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 1)
    C = gb_num2cell (0, A) ;
else
    C = gb_num2cell (0, A, dim) ;
end

