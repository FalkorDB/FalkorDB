function C = gb_imag (ghb, G)
%GB_IMAG implements GrB/imag and GhB/imag.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (G))
    G = struct (G) ;
end

[m, n, type] = gbmex_size (G) ;

if (gb_contains (type, 'complex'))
    % C = imag (G) where G is complex
    C = gzb_apply (ghb, 'cimag', G) ;
else
    % G is real, so C = zeros (m,n)
    C = gzb (ghb, m, n, type) ;
end

