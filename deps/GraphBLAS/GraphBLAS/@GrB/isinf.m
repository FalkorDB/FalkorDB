function C = isinf (G)
%ISINF true for infinite elements.
% C = isinf (G) returns a logical matrix C where C(i,j) = true
% if G(i,j) is infinite.
%
% See also GrB/isnan, GrB/isfinite.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

G = G.opaque ;
[m, n, type] = gbsize (G) ;

if (gb_isfloat (type) && gbnvals (G) > 0)
    C = GrB (gbapply ('isinf', G)) ;
else
    % C is all false
    C = GrB (m, n, 'logical') ;
end

