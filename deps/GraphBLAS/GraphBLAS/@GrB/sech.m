function C = sech (G)
%SECH hyperbolic secant.
% C = sech (G) is the hyperbolic secant of each entry of G.
% Since sech(0) is nonzero, C is a full matrix.
%
% See also GrB/sec, GrB/asec, GrB/asech.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

G = G.opaque ;
type = gbtype (G) ;
if (~gb_isfloat (type))
    type = 'double' ;
end

C = GrB (gbapply ('minv', gbapply ('cosh', gbfull (G, type)))) ;

