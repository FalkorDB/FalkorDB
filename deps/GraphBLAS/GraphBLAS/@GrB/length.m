function n = length (G)
%LENGTH the length of a vector.
% length (G) is the length of the vector G.  For matrices, it is
% max (size (G)) if G is non-empty, or zero if G has any zero dimension.
% If any dimension of G exceeds flintmax, the result is returned as int64
% to avoid integer overflow.
%
% See also GrB/size, GrB/numel.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

G = G.opaque ;
[m, n] = gbsize (G) ;

if (m == 0 || n == 0)
    n = 0 ;
else
    n = max (m, n) ;
end

