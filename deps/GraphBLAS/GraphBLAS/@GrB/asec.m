function C = asec (G)
%ASEC inverse secant.
% C = asec (G) is the inverse secant of each entry of G.
% Since asec (0) is nonzero, the result is a full matrix.
% C is complex if any (abs(G) < 1).
%
% See also GrB/sec, GrB/sech, GrB/asech.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

G = G.opaque ;
type = gbtype (G) ;
if (~gb_isfloat (type))
    type = 'double' ;
end

C = GrB (gb_trig ('acos', gbapply ('minv', gbfull (G, type)))) ;

