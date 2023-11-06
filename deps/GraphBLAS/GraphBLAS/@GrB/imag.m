function C = imag (G)
%IMAG complex imaginary part.
% C = imag (G) returns the imaginary part of G.
%
% See also GrB/conj, GrB/real.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

G = G.opaque ;
[m, n, type] = gbsize (G) ;

if (gb_contains (type, 'complex'))
    % C = imag (G) where G is complex
    C = GrB (gbapply ('cimag', G)) ;
else
    % G is real, so C = zeros (m,n)
    C = GrB (gbnew (m, n, type)) ;
end

