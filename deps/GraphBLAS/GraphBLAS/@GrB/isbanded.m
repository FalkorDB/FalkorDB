function s = isbanded (A, lo, hi)
%ISBANDED true if A is a banded matrix.
% isbanded (A, lo, hi) is true if the bandwidth of A is between lo and hi.
%
% See also GrB/istril, GrB/istriu, GrB/bandwidth.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (A))
    A = struct (A) ;
end

lo = gb_get_scalar (lo) ;
hi = gb_get_scalar (hi) ;

[alo, ahi] = gbmex_bandwidth (A, 1, 1) ;
s = (alo <= lo) & (ahi <= hi) ;

