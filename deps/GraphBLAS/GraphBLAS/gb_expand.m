function C = gb_expand (ghb, scalar, S, type)
%GB_EXPAND expand a scalar into a GraphBLAS matrix.  Not user-callable.
% Implements C = GrB.expand (scalar, S, type).  This function assumes the
% first input is a scalar; the caller has checked this already.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (S))
    S = struct (S) ;
end

if (gb_is_grb (scalar))
    scalar = struct (scalar) ;
end

if (~gb_isscalar (scalar))
    error ('GrB:error', 'first input must be a scalar') ;
end

if (nargin < 4)
    type = gbmex_type (scalar) ;
end

% typecast the scalar to the desired type, and make sure it's full
t = gzb_full (1, gzb (1, scalar, type)) ;

% expand the scalar into the pattern of S
C = gzb_apply2 (ghb, ['2nd.' type], S, t) ;

