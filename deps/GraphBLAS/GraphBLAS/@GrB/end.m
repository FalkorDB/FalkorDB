function i = end (G, k, ndims)
%END Last index in an indexing expression for a GraphBLAS matrix.
%
% See also GrB/size, GrB/length.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (ndims == 1)
    if (isvector (G))
        % G(end) of a vector G
        i = length (G) ;
    else
        % G(end) of a matrix G, for linear indexing
        i = numel (G) ;
    end
elseif (ndims == 2)
    s = size (G) ;
    i = s (k) ;
else
    % sparse N-dimensional arrays for N > 2 will not be supported
    error ('GrB:error', '%dD indexing not supported', ndims) ;
end

