function C = cell2mat (A)
%GRB.CELL2MAT Concatenate a cell array of matrices into a single matrix.
% C = GrB.cell2mat (A) converts a 2D cell array of matrices into a single GrB
% matrix.  Let [m,n] = size(A) be the size of the cell array A.  Then C is
% computed as:
%
%  C = [ A{0,0}   A{0,1}   A{0,2}   ... A{0,n-1}
%        A{1,0}   A{1,1}   A{1,2}   ... A{1,n-1}
%        ...
%        A{m-1,0} A{m-1,1} A{m-1,2} ... A{m-1,n-1} ]
%
% If the matrices in A have different types, the type is determined according
% to the rules in GrB.optype.
%
% Note: The methods in the "cat" family include horzcat, vertcat, cat, cell2mat
% (this method), mat2cell, and num2cell.  All of them appear in GrB, and all
% but this one are overloaded methods.  GrB.cell2mat is a static method, since
% its input is a cell array, not a GrB object, and thus its use cannot trigger
% the call to an overloaded method.  GrB.cell2mat method can operate on any mix
% of GrB/GhB/built-in matrices, with any mix of data types.  The output is
% always a GrB matrix.
%
% This method predates MATLAB R2025a.  MATLAB R2025a and later now allow mixing
% of data types, but the rules for the type of C differ from GrB.cell2mat.  GrB
% selects the largest type of its inputs, while the MATLAB cell2mat selects the
% smallest.
%
% Example:
%
%   A = { [1] [2 3 4] ; [5 ; 9] [6 7 8 ; 10 11 12] } ;
%   C1 = cell2mat (A)
%   C2 = GrB.cell2mat (A)
%   C3 = [ A{1,1} A{1,2} ; A{2,1} A{2,2} ]
%   assert (isequal (C1, C2))
%   assert (isequal (C1, C3))
%
%   % mixing data types: C4 will be double, but C5 will be single.
%   A{1,1} = GrB (1, 'single')
%   for k = 1:numel (A)
%       fprintf ('A {%d} is class: %s, type: %s\n', k, ...
%           class (A {k}), GrB.type (A {k})) ;
%   end
%   C4 = GrB.cell2mat (A)
%   A{1,1} = single (A {1,1}) ;
%   C5 = cell2mat (A)
%   assert (isequal (GrB.type (C4), 'double'))
%   assert (isequal (class (C5), 'single'))
%   assert (isequal (C4, C1)) ;
%   assert (isequal (C5, single (C1))) ;
%
% See also GrB/horzcat, GrB/vertcat, GrB/cat, GrB/mat2cell, GrB/num2cell.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_cell2mat (0, A) ;

