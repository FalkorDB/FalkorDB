function C = gb_single (G)
%GB_SINGLE implements GrB/single and GhB/single.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% FUTURE: exploit single and single complex matrices in MATLAB 2025a.

if (gb_is_grb (G))
    G = struct (G) ;
end

desc.kind = 'builtin' ;
if (gb_contains (gbmex_type (G), 'complex'))
    z = complex (single (0)) ;
    ctype = 'single complex' ;
else
    z = single (0) ;
    ctype = 'single' ;
end

% export C as a full matrix
C = gb_builtin (gzb_full (1, G, ctype, z, desc)) ;

