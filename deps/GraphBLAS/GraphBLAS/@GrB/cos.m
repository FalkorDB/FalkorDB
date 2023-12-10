function C = cos (G)
%COS cosine.
% C = cos (G) is the cosine of each entry of G.
% Since cos (0) = 1, the result is a full matrix.
%
% See also GrB/acos, GrB/cosh, GrB/acosh.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

G = G.opaque ;
type = gbtype (G) ;
if (~gb_isfloat (type))
    type = 'double' ;
end

C = GrB (gbapply ('cos', gbfull (G, type))) ;

