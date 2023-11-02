function C = gb_check_imag_zero (G)
%GB_TO_REAL_IF_IMAG_ZERO convert complex matrix to real if imag(G) is zero

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_contains (gbtype (G), 'complex') && ...
    gbnvals (gbselect ('nonzero', gbapply ('cimag', G))) == 0)
    C = gbapply ('creal', G) ;
else
    C = G ;
end

