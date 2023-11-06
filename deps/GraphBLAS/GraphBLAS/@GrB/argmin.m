function [x,p] = argmin (A, dim)
%GRB.ARGMIN argmin of a built-in or GraphBLAS matrix
%
% [x,p] = argmin(A,1) computes the argmin of each column of A, similar to
%       [x,p] = min(A), or min(A,[],1), with x and p being n-by-1 vectors
%       where A is m-by-n.  If x(j) and p(j) are present, then x(j) =
%       min(A(:,j)) = A(p(j),j).
%
% [x,p] = argmin(A,2) computes the argmin of each row of A, with x and p
%       being m-by-1 vectors where A is m-by-n.  If x(i) and p(i) are
%       present, then x(i) = min(A(i,:)) = A(i,p(i)).
%
% [x,p] = argmin(A,0) computes the argmin of all of A, where x is a scalar,
%       and p is a 2-by-1 vector, with x = min(A,[],'all') = A(p(1),p(2)).
%       If dim is not present, it defaults to 0.
%
% Unlike the built-in min, entries not present in A are not assumed to
% have the value zero.  Instead, they are ignored.  If column A(:,j) has
% no entries, x(j) and p(j) are not present in the sparsity pattern of x
% and p, respectively.  GrB.argmin always returns x and p as column
% vectors, while the built-in min returns x and p as either row or column
% or vectors, depending on dim.  NaNs are ignored.  If x(j) is NaN, p(j)
% is empty.
%
% Example:
%
%   % these produce the same results since no row/column of A is empty
%   A = [ 1 4 9 ; 2 -2 2 ; 3 -10 0 ; 5 4 3 ]
%   [x,p] = GrB.argmin (A,2)
%   [x,p] = min (A, [ ], 2)
%   [x,p] = GrB.argmin (A,1)
%   [x,p] = min (A, [ ], 1)
%
%   % min and GrB.argmin differ since A has an empty row and column
%   A (:,1) = 0 ;
%   A (2,:) = 0 ;
%   A = sparse (A)
%   [x,p] = GrB.argmin (A,2)
%   [x,p] = min (A, [ ], 2)
%   [x,p] = GrB.argmin (A,1)
%   [x,p] = min (A, [ ], 1)
%
%   % the global min of A
%   x = min (A, [ ], 'all')
%   [x,p] = GrB.argmin (A)
%
% Complex matrices are not supported.  If A(i,j) is NaN, then x(i) is NaN
% for argmin(A,2) and p(i) is not present, and x(j) is NaN for argmin(A,1)
% and p(j) is not present.
%
% See also min, max, GrB/min, GrB/max, GrB.argmax, GrB.argsort.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (isobject (A))
    A = A.opaque ;
end

if (nargin < 2)
    dim = 0 ;
end

[x,p] = gbargminmax (A, 0, dim) ;
x = GrB (x) ;
p = GrB (p) ;

