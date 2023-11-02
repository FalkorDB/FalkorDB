function C = acot (G)
%ACOT inverse cotangent.
% C = acot (G) is the inverse cotangent of each entry of G.
% Since acot (0) is nonzero, C is a full matrix.
%
% See also GrB/cot, GrB/coth, GrB/acoth.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

G = G.opaque ;
type = gbtype (G) ;

if (~gb_isfloat (type))
    type = 'double' ;
end

C = GrB (gbapply ('atan', gbapply ('minv', gbfull (G, type)))) ;

