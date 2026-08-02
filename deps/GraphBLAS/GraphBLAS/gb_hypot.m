function C = gb_hypot (ghb, A_arg, B_arg)
%GB_HYPOT implements hypot for GrB and GhB.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (A_arg))
    A_arg = struct (A_arg) ;
end

if (gb_is_grb (B_arg))
    B_arg = struct (B_arg) ;
end

atype = gbmex_type (A_arg) ;
btype = gbmex_type (B_arg) ;

if (gb_contains (atype, 'complex'))
    A = gzb_apply (1, 'abs', A_arg) ;
elseif (~gb_isfloat (atype))
    A = gzb (1, A_arg, 'double') ;
else
    % use A_arg as-is
    A = A_arg ;
end

if (gb_contains (btype, 'complex'))
    B = gzb_apply (1, 'abs', B_arg) ;
elseif (~gb_isfloat (btype))
    B = gzb (1, B_arg, 'double') ;
else
    % use B_arg as-is
    B = B_arg ;
end

C = gzb_apply (ghb, 'abs', gb_eadd (1, A, 'hypot', B)) ;

