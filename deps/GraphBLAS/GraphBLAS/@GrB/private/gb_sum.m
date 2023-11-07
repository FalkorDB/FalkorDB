function C = gb_sum (op, G, option)
%GB_SUM C = sum (G) or C = any (G)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 2)
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
        C = gbreduce (op, G) ;

    case { 1 }

        % C = sum (G, 1) reduces each column to a scalar,
        % giving a 1-by-n row vector.
        desc.in0 = 'transpose' ;
        C = gbtrans (gbvreduce (op, G, desc)) ;

    case { 2 }

        % C = sum (G, 2) reduces each row to a scalar,
        % giving an m-by-1 column vector.
        C = gbvreduce (op, G) ;

    otherwise

        error ('GrB:error', 'unknown option') ;
end

