function s = istril (G)
%ISTRIL determine if a matrix is lower triangular.
% istril (G) is true if all entries in G are on or below the diagonal.  A
% GraphBLAS matrix G may have explicit zeros.  If these appear in the upper
% triangular part of G, then istril (G) is false, but istril (double (G))
% can be true since double (G) drops those entries.
%
% See also GrB/istriu, GrB/isbanded.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[~,hi] = gbbandwidth (G.opaque, 0, 1) ;
s = (hi == 0) ;

