function C = cell2mat (A)
%GRB.CELL2MAT Concatenate a cell array of matrices into a single matrix.
% C = GrB.cell2mat (A) converts a 2D cell array of matrices into a single
% @GrB matrix.  The input matrices in A may be @GrB/built-in
% matrices, in any combination.  Let [m,n] = size(A) be the size of the
% cell array A.  Then C is computed as:
%
%  C = [ A{0,0}   A{0,1}   A{0,2}   ... A{0,n-1}
%        A{1,0}   A{1,1}   A{1,2}   ... A{1,n-1}
%        ...
%        A{m-1,0} A{m-1,1} A{m-1,2} ... A{m-1,n-1} ]
%
% If the matrices in A have different types, the type is determined
% according to the rules in GrB.optype.
%
% Note: The methods in the "cat" family include horzcat, vertcat, cat,
% cell2mat (this method), mat2cell, and num2cell.  All of them appear in
% @GrB, and all but this one are overloaded methods.  GrB.cell2mat is a
% static method, since its input is a cell array, not a @GrB object, and
% thus its use cannot trigger the call to an overloaded method.  The
% built-in cell2mat method fails if any A{i,j} entry is a @GrB
% matrix, and it also requires that all entries in A have the same data
% type.  GrB.cell2mat method can operate on any mix of @GrB/built-in
% matrices, with any mix of data types.
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
%   % mixing data types:
%   A{1,1} = GrB (1, 'single')
%   for k = 1:numel (A)
%       fprintf ('A {%d} is class: %s, type: %s\n', k, ...
%           class (A {k}), GrB.type (A {k})) ;
%   end
%   C2 = GrB.cell2mat (A)
%   assert (isequal (C1, C2))
%   A{1,1} = single (1)
%   A
%   for k = 1:numel (A)
%       fprintf ('A {%d} is class: %s\n', k, class (A {k})) ;
%   end
%   try
%       % the built-in cell2mat does not support mixing of data types.
%       C = cell2mat (A)
%   catch me
%       fprintf ('expected error:\n') ;
%       me
%   end
%
% See also GrB/horzcat, GrB/vertcat, GrB/cat, GrB/mat2cell, GrB/num2cell.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (~iscell (A))
    error ('GrB:error', 'input must be a cell array') ;
end
if (ndims (A) > 2) %#ok<ISMAT>
    error ('GrB:error', 'only 2D cell arrays are supported') ;
end

% get the input matrices
[m, n] = size (A) ;
for j = 1:n
    for i = 1:m
        Tile = A {i,j} ;
        if (isobject (Tile))
            A {i,j} = Tile.opaque ;
        end
    end
end

% concatenate the matrices
C = GrB (gbcat (A)) ;

