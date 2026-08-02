function s = gb_issymmetric (G_arg, option, herm)
%GB_ISSYMMETRIC check if symmetric or Hermitian.  Not user-callable.
% Implements issymmetric (G,option) and ishermitian (G,option).

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% FUTURE: this can be much faster; see spsym in CHOLMOD.

if (gb_is_grb (G_arg))
    G_arg = struct (G_arg) ;
end

[m, n, type] = gbmex_size (G_arg) ;

if (m ~= n)

    s = false ;

else

    if (isequal (type, 'logical'))
        G = gzb (1, G_arg, 'double') ;
    else
        G = G_arg ;
    end

    if (herm && gb_contains (type, 'complex'))
        % T = G', complex conjugate transpose
        desc.in0 = 'transpose' ;
        T = gzb_apply (1, 'conj', G, desc) ;
    else
        % T = G.', array transpose
        T = gzb_trans (1, G) ;
    end

    switch (option)

        case { 'skew' }

            % G is skew symmetric/Hermitian if G+T is zero
            s = (gzb_norm (gb_eadd (1, G, '+', T), 1) == 0) ;

        case { 'nonskew' }

            % G is symmetric/Hermitian if G-T is zero
            s = (gzb_normdiff (G, T, 1) == 0) ;

        otherwise

            error ('GrB:error', 'invalid option') ;

    end

    if (s)
        % also check the pattern; G might have explicit zeros
        S = gb_spones (1, G, 'logical') ;
        T = gzb_trans (1, S) ;
        s = gb_isequal (S, T) ;
    end
end

