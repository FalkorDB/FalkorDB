function C = gb_spfun (ghb, fun, G)
%GB_SPFUN implements GrB/spfun and GhB/spfun.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (G))
    G = struct (G) ;
end

if (ischar (fun))
    try
        C = gzb_apply (ghb, fun, G) ;
        return ;
    catch me %#ok<NASGU>
        % gzb_apply failed; fall through to feval below
    end
end

% 'fun' is not a string, or not a built-in GraphBLAS operator
[m, n] = gbmex_size (G) ;
desc.base = 'zero-based' ;
gbmex_wait (G) ;
[I, J, X] = gbmex_extracttuples (1, G, desc) ;
X = feval (fun, X) ;
C = gzb_build (ghb, I, J, X, m, n, '1st', desc) ;

