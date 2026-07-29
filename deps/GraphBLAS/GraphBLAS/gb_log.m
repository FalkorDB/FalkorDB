function C = gb_log (ghb, G)
%GB_LOG implements GrB/log and GhB/log.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (G))
    G = struct (G) ;
end

C = gb_trig (ghb, 'log', gzb_full (1, G)) ;

if (gb_make_real (C))
    C = gzb_apply (ghb, 'creal', C) ;
end

