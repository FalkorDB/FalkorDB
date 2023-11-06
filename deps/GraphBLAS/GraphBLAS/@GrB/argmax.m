function [x,p] = argmax (A, dim)
%GRB.ARGMIN argmax of a built-in or GraphBLAS matrix
%
% [x,p] = argmax(A,1) computes the argmax of each column of A, similar to
%       [x,p] = max(A), or max(A,[],1), with x and p being n-by-1 vectors
%       where A is m-by-n.  If x(j) and p(j) are present, then x(j) =
%       max(A(:,j)) = A(p(j),j).
%
% [x,p] = argmax(A,2) computes the argmax of each row of A, with x and p
%       being m-by-1 vectors where A is m-by-n.  If x(i) and p(i) are
%       present, then x(i) = max(A(i,:)) = A(i,p(i)).
%
% [x,p] = argmax(A,0) computes the argmax of all of A, where x is a scalar,
%       and p is a 2-by-1 vector, with x = max(A,[],'all') = A(p(1),p(2)).
%       If dim is not present, it defaults to 0.
%
% Unlike the built-in max, entries not present in A are not assumed
% to have the value zero.  Instead, they are ignored.  If column A(:,j)
% has no entries, x(j) and p(j) are not present in the sparsity pattern of
% x and p, respectively.  GrB.argmax always returns x and p as column
% vectors, while the built-in max returns x and p as either row or column
% or vectors, depending on dim.  NaNs are ignored.  If x(j) is NaN, p(j)
% is empty.
%
% Example:
%
%   % these produce the same results since no row/column of A is empty
%   A = [ 1 4 9 ; 2 -2 2 ; 3 -10 0 ; 5 4 3 ]
%   [x,p] = GrB.argmax (A,2)
%   [x,p] = max (A, [ ], 2)
%   [x,p] = GrB.argmax (A,1)
%   [x,p] = max (A, [ ], 1)
%
%   % max and GrB.argmax differ since A has an empty row and column
%   A (:,1) = 0 ;
%   A (2,:) = 0 ;
%   A = sparse (A)
%   [x,p] = GrB.argmax (A,2)
%   [x,p] = max (A, [ ], 2)
%   [x,p] = GrB.argmax (A,1)
%   [x,p] = max (A, [ ], 1)
%
%   % the global max of A
%   x = max (A, [ ], 'all')
%   [x,p] = GrB.argmax (A)
%
% Complex matrices are not supported.  If A(i,j) is NaN, then x(i) is NaN
% for argmax(A,2) and p(i) is not present, and x(j) is NaN for argmax(A,1)
% and p(j) is not present.
%
% See also min, max, GrB/min, GrB/max, GrB.argmin, GrB.argsort.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (isobject (A))
    A = A.opaque ;
end

if (nargin < 2)
    dim = 0 ;
end

[x,p] = gbargminmax (A, 1, dim) ;
x = GrB (x) ;
p = GrB (p) ;

