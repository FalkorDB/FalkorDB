function C = cbrt (G)
%CBRT cube root
% C = cbrt (G) is the cube root of the entries of G.
%
% See also GrB/sqrt, nthroot.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

G = G.opaque ;
type = gbtype (G) ;
if (gb_contains (type, 'complex'))
    error ('GrB:error', 'input must be real') ;
elseif (gb_isfloat (type))
    op = 'cbrt' ;
else
    op = 'cbrt.double' ;
end

C = GrB (gbapply (op, G)) ;

