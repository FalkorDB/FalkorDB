function C = any (G, option)
%ANY true if any element of a matrix is nonzero or true.
% C = any (G) is true if any entry in G is nonzero or true.  If G is a
% matrix, C is a row vector with C(j) = any (G (:,j)).
%
% C = any (G, 'all') is a scalar, true if any entry in G is nonzero or true
% C = any (G, 1) is a row vector with C(j) = any (G (:,j))
% C = any (G, 2) is a column vector with C(i) = any (G (i,:))
%
% See also GhB/all, GhB/sum, GhB/nnz, GhB.entries, GhB.nonz.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 1)
    C = gb_sum (1, '|.logical', 'logical', G) ;
else
    C = gb_sum (1, '|.logical', 'logical', G, option) ;
end

