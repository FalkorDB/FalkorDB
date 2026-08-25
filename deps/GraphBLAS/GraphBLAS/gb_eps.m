function C = gb_eps (ghb, G)
%GB_EPS implements GrB/eps and GhB/eps.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% FUTURE: GraphBLAS should have a built-in eps unary operator.

% FUTURE: there should be a sparse version of 'eps'.
% C is full because eps (0) is 2^(-1024).

if (gb_is_grb (G))
    G = struct (G) ;
end

% convert to a built-in full matrix and use the built-in eps
switch (gbmex_type (G))

    case { 'single' }
        T = eps (gb_single (gb_full (1, G))) ;

    case { 'double' }
        T = eps (gb_double (gb_full (1, G))) ;

    case { 'single complex' }
        T = max (eps (gb_single (gb_real (1, G))), ...
                 eps (gb_single (gb_imag (1, G)))) ;

    case { 'double complex' }
        T = max (eps (gb_double (gb_real (1, G))), ...
                 eps (gb_double (gb_imag (1, G)))) ;

    otherwise
        error ('GrB:error', 'input must be floating-point') ;

end

C = gb_dup (ghb, T) ;

