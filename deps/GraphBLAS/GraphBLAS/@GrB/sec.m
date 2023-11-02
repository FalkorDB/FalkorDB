function C = sec (G)
%SEC secant.
% C = sec (G) is the secant of each entry of G.
% Since sec (0) = 1, the result is a full matrix.
%
% See also GrB/asec, GrB/sech, GrB/asech.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

G = G.opaque ;
type = gbtype (G) ;
if (~gb_isfloat (type))
    type = 'double' ;
end

C = GrB (gbapply ('minv', gbapply ('cos', gbfull (G, type)))) ;

