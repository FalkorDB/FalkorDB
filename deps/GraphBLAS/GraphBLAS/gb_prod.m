function C = gb_prod (ghb, op, type, G, option)
%GB_PROD C = prod (G), using the given operator and type.  Not user-callable.
% Implements C = prod (G) and C = all (G).

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (G))
    G = struct (G) ;
end

if (isempty (type))
    type = gbmex_type (G) ;
end

if (isequal (op, '*') && isequal (type, 'logical'))
    % revise the op for the *.logical case
    op = '&.logical' ;
end

[m, n] = gbmex_size (G) ;

if (nargin < 5)
    % C = prod (G)
    if (m == 1 || n == 1)
        option = 'all' ;
    else
        option = 1 ;
    end
end

switch (option)

    case { 'all' }

        % C = prod (G, 'all'), reducing all entries to a scalar
        if (gb_isfull (G))
            C = gzb_reduce (ghb, op, G) ;
        else
            C = gzb (ghb, 0, type) ;
        end

    case { 1 }

        % C = prod (G,1) reduces each column to a scalar,
        % giving a 1-by-n row vector.
        % M = find (column degree of G == m)
        M = gzb_select (1, gzb_degree (1, G, 'col'), '==', int64 (m)) ;
        Cin = gzb (1, n, 1, type) ;
        % C<M> = op (G.')
        desc.in0 = 'transpose' ;
        GT = gzb_vreduce (1, Cin, M, G, op, desc) ;
        C = gzb_trans (ghb, GT) ;

    case { 2 }

        % C = prod (G,2) reduces each row to a scalar,
        % giving an m-by-1 column vector.
        % M = find (row degree of G == n)
        M = gzb_select (1, gzb_degree (1, G, 'row'), '==', int64 (n)) ;
        % C<M> = op (G)
        Cin = gzb (1, m, 1, type) ;
        C = gzb_vreduce (ghb, Cin, M, G, op) ;

    otherwise

        error ('GrB:error', 'unknown option') ;
end

