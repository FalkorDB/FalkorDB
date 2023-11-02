function U = triu (G, k)
%TRIU upper triangular part of a matrix.
% U = triu (G) returns the upper triangular part of G.
%
% U = triu (G,k) returns the entries on and above the kth diagonal of X,
% where k=0 is the main diagonal.
%
% See also GrB/tril.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (isobject (G))
    G = G.opaque ;
end

if (nargin < 2)
    k = 0 ;
else
    k = gb_get_scalar (k) ;
end

U = GrB (gbselect ('triu', G, k)) ;

