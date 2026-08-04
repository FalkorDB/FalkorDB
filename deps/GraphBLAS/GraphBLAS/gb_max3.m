function C = gb_max3 (ghb, op, A, option)
%GB_MAX3 3-input max.  Not user-callable.
% Implements C = max (A, [ ], option)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (isequal (option, 'all'))
    % C = max (A, [ ] 'all'), reducing all entries to a scalar
    C = gb_maxall (ghb, op, A) ;
else
    opt = gb_get_scalar (option) ;
    if (opt == 1)
        % C = max (A, [ ], 1) reduces each column to a scalar,
        % giving a 1-by-n row vector.
        C = gb_maxbycol (ghb, op, A) ;
    elseif (opt == 2)
        % C = max (A, [ ], 2) reduces each row to a scalar,
        % giving an m-by-1 column vector.
        C = gb_maxbyrow (ghb, op, A) ;
    else
        error ('GrB:error', 'invalid option') ;
    end
end

