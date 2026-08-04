function C = all (G, option)
%ALL True if all elements of a GraphBLAS matrix are nonzero or true.
% C = all (G) is true if all entries in G are nonzero or true.  If G is a
% matrix, C is a row vector with C(j) = all (G (:,j)).
%
% C = all (G, 'all') is a scalar, true if all entries G are nonzero or true
% C = all (G, 1) is a row vector with C(j) = all (G (:,j))
% C = all (G, 2) is a column vector with C(i) = all (G (i,:))
%
% See also GhB/any, GhB/nnz, GhB/prod, GhB.entries.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 1)
    C = gb_prod (1, '&.logical', 'logical', G) ;
else
    C = gb_prod (1, '&.logical', 'logical', G, option) ;
end

