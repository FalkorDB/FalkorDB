function s = gb_issymmetric (G, option, herm)
%GB_ISSYMMETRIC check if symmetric or Hermitian
% Implements issymmetric (G,option) and ishermitian (G,option).

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% FUTURE: this can be much faster; see spsym in CHOLMOD.

[m, n, type] = gbsize (G) ;

if (m ~= n)

    s = false ;

else

    if (isequal (type, 'logical'))
        G = gbnew (G, 'double') ;
    end

    if (herm && gb_contains (type, 'complex'))
        % T = G', complex conjugate transpose
        desc.in0 = 'transpose' ;
        T = gbapply ('conj', G, desc) ;
    else
        % T = G.', array transpose
        T = gbtrans (G) ;
    end

    switch (option)

        case { 'skew' }

            % G is skew symmetric/Hermitian if G+T is zero
            s = (gbnorm (gb_eadd (G, '+', T), 1) == 0) ;

        case { 'nonskew' }

            % G is symmetric/Hermitian if G-T is zero
            s = (gbnormdiff (G, T, 1) == 0) ;

        otherwise

            error ('GrB:error', 'invalid option') ;

    end

    clear T

    if (s)
        % also check the pattern; G might have explicit zeros
        S = gb_spones (G, 'logical') ;
        s = gbisequal (S, gbtrans (S)) ;
    end
end

