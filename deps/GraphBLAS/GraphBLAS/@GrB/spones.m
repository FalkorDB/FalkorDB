function C = spones (G, type)
%SPONES return pattern of a sparse matrix.
% C = spones (G) returns a matrix C with the same pattern as G, but with
% all entries set to 1.  The behavior of spones (G) for a GrB matrix
% differs from spones (A) for a built-in matrix A.  An explicit entry
% G(i,j) that has a value of zero is converted to the explicit entry
% C(i,j)=1.  Explicit zero entries never appear in a built-in sparse
% matrix.
%
% C = spones (G) returns C as the same type as G if G is real.
% If G is complex, C has the underlying real type of G ('single' if
% G is 'single complex', or 'double' if G is 'double complex').
%
% C = spones (G,type) returns C in the requested type ('double',
% 'single', 'int8', ...).  For example, use C = spones (G, 'logical') to
% return the pattern of G as a sparse logical matrix.
%
% See also GrB/spfun, GrB.apply.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

G = G.opaque ;
if (nargin == 1)
    C = GrB (gb_spones (G)) ;
else
    C = GrB (gb_spones (G, type)) ;
end

