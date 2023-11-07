function s = isbanded (A, lo, hi)
%ISBANDED true if A is a banded matrix.
% isbanded (A, lo, hi) is true if the bandwidth of A is between lo and hi.
%
% See also GrB/istril, GrB/istriu, GrB/bandwidth.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (isobject (A))
    A = A.opaque ;
end

lo = gb_get_scalar (lo) ;
hi = gb_get_scalar (hi) ;

[alo, ahi] = gbbandwidth (A, 1, 1) ;
s = (alo <= lo) & (ahi <= hi) ;

