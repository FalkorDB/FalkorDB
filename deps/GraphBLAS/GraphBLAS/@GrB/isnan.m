function C = isnan (G)
%ISNAN true for NaN elements.
% C = isnan (G) is a logical C matrix with C(i,j)=true if G(i,j) is NaN.
%
% See also GrB/isinf, GrB/isfinite.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

G = G.opaque ;
[m, n, type] = gbsize (G) ;

if (gb_isfloat (type) && gbnvals (G) > 0)
    C = GrB (gbapply ('isnan', G)) ;
else
    % C is all false
    C = GrB (m, n, 'logical') ;
end

