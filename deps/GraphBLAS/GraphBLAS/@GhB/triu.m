function U = triu (G, k)
%TRIU upper triangular part of a matrix.
% U = triu (G) returns the upper triangular part of G.
%
% U = triu (G,k) returns the entries on and above the kth diagonal of X,
% where k=0 is the main diagonal.
%
% See also GhB/tril.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin < 2)
    U = gb_tri (1, 'triu', G, 0) ;
else
    U = gb_tri (1, 'triu', G, k) ;
end

