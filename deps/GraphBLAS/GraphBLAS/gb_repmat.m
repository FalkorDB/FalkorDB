function C = gb_repmat (ghb, G, m, n)
%GB_REPMAT implements GrB/repmat and GhB/repmat.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (G))
    G = struct (G) ;
end

type = gbmex_type (G) ;
if (nargin == 4)
    R = ones (m, n, 'logical') ;
else
    R = ones (m, 'logical') ;
end
op = ['2nd.' type] ;

C = gzb_kronecker (ghb, R, op, G) ;

