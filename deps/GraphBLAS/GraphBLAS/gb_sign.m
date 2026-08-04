function C = gb_sign (ghb, G)
%GB_SIGN implements GrB/sign and GhB/sign.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (G))
    G = struct (G) ;
end

type = gbmex_type (G) ;

if (isequal (type, 'logical'))
    C = gb_dup (ghb, G) ;
elseif (~gb_isfloat (type))
    T = gzb_apply (1, 'signum.single', G) ;
    C = gzb (ghb, T, type) ;
else
    C = gzb_apply (ghb, 'signum', G) ;
end

