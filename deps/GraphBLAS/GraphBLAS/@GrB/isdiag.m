function s = isdiag (G)
%ISDIAG true if G is a diagonal matrix.
% isdiag (G) is true if G is a diagonal matrix, and false otherwise.
%
% See also GrB/isbanded.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[lo,hi] = gbbandwidth (G.opaque, 1, 1) ;
s = (lo == 0) && (hi == 0) ;

