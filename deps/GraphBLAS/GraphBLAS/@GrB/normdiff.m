function s = normdiff (A,B,kind)
%NORMDIFF norm (A-B,kind)
% If A-B is a matrix:
%
%   norm (A-B,1) is the maximum sum of the columns of abs (A-B).
%   norm (A-B,inf) is the maximum sum of the rows of abs (A-B).
%   norm (A-B,'fro') is the Frobenius norm of A-B: the sqrt of the sum of
%       the squares of the entries in A-B.
%   The 2-norm is not available for either built-in or GraphBLAS sparse
%       matrices.
%
% If A-B is a row or column vector:
%
%   norm (A-B,1) is the sum of abs (A-B)
%   norm (A-B,2) is the sqrt of the sum of (A-B).^2
%   norm (A-B,inf) is the maximum of abs (A-B)
%   norm (A-B,-inf) is the minimum of abs (A-B)
%
% See also GrB.reduce, GrB/norm.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin < 3)
    kind = 2 ;
end

if (isobject (A))
    A = A.opaque ;
end

if (isobject (B))
    B = B.opaque ;
end

s = gbnormdiff (A, B, kind) ;

