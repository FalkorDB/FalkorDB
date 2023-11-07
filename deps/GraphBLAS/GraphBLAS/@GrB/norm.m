function s = norm (G, kind)
%NORM matrix or vector norm.
% If G is a matrix:
%
%   norm (G,1) is the maximum sum of the columns of abs (G).
%   norm (G,inf) is the maximum sum of the rows of abs (G).
%   norm (G,'fro') is the Frobenius norm of G: the sqrt of the sum of the
%       squares of the entries in G.
%   The 2-norm is not available for either built-in or GraphBLAS sparse
%       matrices.
%
% If G is a row or column vector:
%
%   norm (G,1) is the sum of abs (G)
%   norm (G,2) is the sqrt of the sum of G.^2
%   norm (G,inf) is the maximum of abs (G)
%   norm (G,-inf) is the minimum of abs (G)
%
% The p-norm for vectors is not yet supported.
%
% See also GrB.reduce, GrB.normdiff.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% FUTURE: add the p-norm for vectors.

if (isobject (G))
    G = G.opaque ;
end

if (nargin == 2)
    if (~ischar (kind))
        kind = gb_get_scalar (kind) ;
    end
else
    kind = 2 ;
end

s = gbnorm (G, kind) ;

