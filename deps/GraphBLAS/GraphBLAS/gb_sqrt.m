function C = gb_sqrt (ghb, G)
%GB_SQRT implements GrB/sqrt and GhB/sqrt.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (G))
    G = struct (G) ;
end

C = gb_trig (ghb, 'sqrt', G) ;

if (gb_make_real (C))
    C = gzb_apply (ghb, 'creal', C) ;
end

