function C = gb_log10 (ghb, G)
%GB_LOG10 implements GrB/log10 and GhB/log10.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (G))
    G = struct (G) ;
end

C = gb_trig (ghb, 'log10', gzb_full (1, G)) ;

if (gb_make_real (C))
    C = gzb_apply (ghb, 'creal', C) ;
end

