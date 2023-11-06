function C = asech (G)
%ASECH inverse hyperbolic secant.
% C = asech (G) is the inverse hyperbolic secant of each entry of G.
% Since asech (0) is nonzero, the result is a full matrix.  C is complex
% if G is complex, or if any real entries are outside of the range [0,1].
%
% See also GrB/sec, GrB/asec, GrB/sech.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

G = G.opaque ;
type = gbtype (G) ;
if (~gb_isfloat (type))
    type = 'double' ;
end

C = GrB (gb_trig ('acosh', gbapply ('minv', gbfull (G, type)))) ;

