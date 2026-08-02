function C = gb_sum (ghb, op, type, G, option)
%GB_SUM C = sum (G) or C = any (G).  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (G))
    G = struct (G) ;
end

if (isempty (type))
    type = gbmex_type (G) ;
end

if (isequal (op, '+') && isequal (type, 'logical'))
    % revise the op for the +.logical case
    op = '+.int64' ;
end

if (nargin < 5)
    % C = sum (G)
    if (gb_isvector (G))
        option = 'all' ;
    else
        option = 1 ;
    end
end

switch (option)

    case { 'all' }

        % C = sum (G, 'all'), reducing all entries to a scalar
        C = gzb_reduce (ghb, op, G) ;

    case { 1 }

        % C = sum (G, 1) reduces each column to a scalar,
        % giving a 1-by-n row vector.
        desc.in0 = 'transpose' ;
        C = gzb_trans (ghb, gzb_vreduce (1, G, op, desc)) ;

    case { 2 }

        % C = sum (G, 2) reduces each row to a scalar,
        % giving an m-by-1 column vector.
        C = gzb_vreduce (ghb, G, op) ;

    otherwise

        error ('GrB:error', 'unknown option') ;
end

