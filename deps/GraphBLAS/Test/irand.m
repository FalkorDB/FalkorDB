function I = irand (imin, imax, m, n)
%IRAND construct a random integer matrix 
%
% return a random m-by-n matrix of integers (uint64)
% in the range imin:imax, inclusive
%
% I = irand (imin, imax, m, n)
%
% if imin > imax, the ranges are swapped.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin < 4)
    n = 1 ;
end

if (nargin < 3)
    m = 1 ;
end

if (imin > imax)
    t = imin ;
    imin = imax ;
    imax = t ;
end

if (imin == imax)
    I = uint64 (imin) * ones (m, n, 'uint64') ;
else
    x = rand (m,n) ;
    I = uint64 (floor ((imax-imin+1) * x) + imin) ;
end

assert (min (min (I)) >= imin) ;
assert (max (max (I)) <= imax) ;


