function C = erfc (G)
%ERFC complementary error function.
% C = erfc (G) is the complementary error function of each entry of G.
% Since erfc (0) = 1, the result is a full matrix.  G must be real.
%
% See also GrB/erf, erfcx, erfinv, erfcinv.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

G = G.opaque ;
type = gbtype (G) ;
if (gb_contains (type, 'complex'))
    error ('GrB:error', 'input must be real') ;
end
if (~gb_isfloat (type))
    type = 'double' ;
end

C = GrB (gbapply ('erfc', gbfull (G, type))) ;

