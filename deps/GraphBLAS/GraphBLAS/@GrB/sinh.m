function C = sinh (G)
%SINH hyperbolic sine.
% C = sinh (G) is the hyperbolic sine of each entry of G.
%
% See also GrB/sin, GrB/asin, GrB/asinh.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

G = G.opaque ;
if (~gb_isfloat (gbtype (G)))
    op = 'sinh.double' ;
else
    op = 'sinh' ;
end

C = GrB (gbapply (op, G)) ;

