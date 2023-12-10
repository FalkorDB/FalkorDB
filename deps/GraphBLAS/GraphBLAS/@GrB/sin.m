function C = sin (G)
%SIN sine.
% C = sin (G) is the sine of each entry of G.
%
% See also GrB/asin, GrB/sinh, GrB/asinh.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

G = G.opaque ;
if (~gb_isfloat (gbtype (G)))
    op = 'sin.double' ;
else
    op = 'sin' ;
end

C = GrB (gbapply (op, G)) ;

