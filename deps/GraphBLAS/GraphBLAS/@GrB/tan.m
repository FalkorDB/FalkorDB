function C = tan (G)
%TAN tangent.
% C = tan (G) is the tangent of each entry of G.
%
% See also GrB/tanh, GrB/atan, GrB/atanh, GrB/atan2.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

G = G.opaque ;
if (~gb_isfloat (gbtype (G)))
    op = 'tan.double' ;
else
    op = 'tan' ;
end

C = GrB (gbapply (op, G)) ;

