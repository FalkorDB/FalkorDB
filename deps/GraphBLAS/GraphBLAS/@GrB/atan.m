function C = atan (G)
%ATAN inverse tangent.
% C = atan (G) is the inverse tangent of each entry of G.
%
% See also GrB/tan, GrB/tanh, GrB/atanh, GrB/atan2.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

G = G.opaque ;

if (~gb_isfloat (gbtype (G)))
    op = 'atan.double' ;
else
    op = 'atan' ;
end

C = GrB (gbapply (op, G)) ;

