function C = gb_make_real (G)
%GB_MAKE_REAL convert complex matrix to real if imag(G) is zero

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_contains (gbtype (G), 'complex') && ...
    gbnvals (gbselect ('nonzero', gbapply ('cimag', G))) == 0)
    C = gbapply ('creal', G) ;
else
    C = G ;
end

