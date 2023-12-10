function C = num2cell (A, dim)
%NUM2CELL Convert matrix into cell array.
%
% C = num2cell (A) converts a @GrB matrix A into a cell array C by
% placing each entry of A in a separate cell, C{i,j} = A(i,j).
% C = num2cell (A, 1) creates a 1-by-n cell array, where A is m-by-n,
% and C{j} is the jth column of A; that is, C{j} = A(:,j).
% C = num2cell (A, 2) creates an m-by-1 cell array where C{i} is the
% ith row of A; that is, C{i} = A (i,:).
% C = num2cell (A, [1 2]) constructs a 1-by-1 cell array C with C{1}=A.
% C = num2cell (A, [2 1]) constructs a 1-by-1 cell array C with C{1}=A.',
% the array transpose of A.
%
% See also GrB/horzcat, GrB/vertcat, GrB/cat, GrB.cell2mat, GrB/mat2cell.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 2 && isequal (dim, [1 2]))

    % whole matrix, not transposed
    C = { A } ;

elseif (nargin == 2 && isequal (dim, [2 1]))

    % whole matrix, transposed
    C = { A.' } ;

else

    % split into scalars, rows, or columns
    if (isobject (A))
        A = A.opaque ;
        [m, n] = gbsize (A) ;
    else
        [m, n] = size (A) ;
    end

    if (nargin == 1)
        % split A into scalars
        C = gbsplit (A, ones (m, 1), ones (n, 1)) ;
    elseif (isequal (dim, 1))
        % split A into columns
        C = gbsplit (A, m, ones (n, 1)) ;
    elseif (isequal (dim, 2))
        % split A into rows
        C = gbsplit (A, ones (m, 1), n) ;
    else
        error ('GrB:error', 'unknown option') ;
    end

    % convert each cell back into GrB matrices
    for k = 1:numel(C)
        C {k} = GrB (C {k}) ;
    end
end

