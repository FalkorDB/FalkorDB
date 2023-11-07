function C = all (G, option)
%ALL True if all elements of a GraphBLAS matrix are nonzero or true.
% C = all (G) is true if all entries in G are nonzero or true.  If G is a
% matrix, C is a row vector with C(j) = all (G (:,j)).
%
% C = all (G, 'all') is a scalar, true if all entries G are nonzero or true
% C = all (G, 1) is a row vector with C(j) = all (G (:,j))
% C = all (G, 2) is a column vector with C(i) = all (G (i,:))
%
% See also GrB/any, GrB/nnz, GrB/prod, GrB.entries.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

G = G.opaque ;

if (nargin == 1)
    C = GrB (gb_prod ('&.logical', 'logical', G)) ;
else
    C = GrB (gb_prod ('&.logical', 'logical', G, option)) ;
end

