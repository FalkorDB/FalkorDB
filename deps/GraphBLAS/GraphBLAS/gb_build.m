function C = gb_build (ghb, I, J, X, varargin) ;
%GB_BUILD implements GrB.build and GhB.build.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

narginchk (4, 9) ;

if (gb_is_grb (I))
    I = struct (I) ;
end

if (gb_is_grb (J))
    J = struct (J) ;
end

if (gb_is_grb (X))
    X = struct (X) ;
end

[C_opaque, kind] = gbmex_build (ghb, I, J, X, varargin {:}) ;

C = gb_mexfunction_result (ghb, C_opaque, kind) ;

