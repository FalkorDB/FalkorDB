function C = isfinite (G)
%ISFINITE true for finite elements.
% C = isfinite (G) is a logical matrix where C(i,j) = true
% if G(i,j) is finite.  C is a full matrix.
%
% See also GrB/isnan, GrB/isinf.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

G = G.opaque ;
[m, n, type] = gbsize (G) ;

if (gb_isfloat (type) && m > 0 && n > 0)
    C = GrB (gbapply ('isfinite', gbfull (G))) ;
else
    % C is all true
    C = GrB (true (m, n)) ;
end

