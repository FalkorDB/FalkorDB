function s = gb_make_real (G)
%GB_MAKE_REAL true if a complex matrix has zero imag part.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (G))
    G = struct (G) ;
end

s = gb_contains (gbmex_type (G), 'complex') && ...
    (gbmex_nvals (gzb_select (1, 'nonzero', gzb_apply (1, 'cimag', G))) == 0) ;

