function C = conj (G)
%CONJ complex conjugate.
% C = conj (G) is the complex conjugate of each entry of G.
%
% See also GrB/real, GrB/imag.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

Q = G.opaque ;

if (gb_contains (gbtype (Q), 'complex'))
    C = GrB (gbapply ('conj', Q)) ;
else
    C = G ;
end

