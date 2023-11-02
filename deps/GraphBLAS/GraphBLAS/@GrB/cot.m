function C = cot (G)
%COT cotangent.
% C = cot (G) is the cotangent of each entry of G.
% Since cot (0) is nonzero, C is a full matrix.
%
% See also GrB/coth, GrB/acot, GrB/acoth.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

G = G.opaque ;
if (~gb_isfloat (gbtype (G)))
    op = 'tan.double' ;
else
    op = 'tan' ;
end

C = GrB (gbapply ('minv', gbfull (gbapply (op, G)))) ;

