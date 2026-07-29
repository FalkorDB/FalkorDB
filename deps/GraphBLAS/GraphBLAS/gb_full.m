function C = gb_full (ghb, A, type, identity)
%GB_FULL implements GrB/full and GhB/full.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (A))
    A = struct (A) ;
end

if (nargin < 3)
    type = gbmex_type (A) ;
end

if (nargin < 4)
    identity = 0 ;
end

% convert A to a full GraphBLAS matrix
C = gzb_full (ghb, A, type, identity) ;

