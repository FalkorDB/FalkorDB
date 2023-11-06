function C = gb_min3 (op, A, option)
%GB_MIN3 3-input min
% Implements C = min (A, [ ], option)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (isequal (option, 'all'))
    % C = min (A, [ ] 'all'), reducing all entries to a scalar
    C = gb_minall (op, A) ;
else
    option = gb_get_scalar (option) ;
    if (option == 1)
        % C = min (A, [ ], 1) reduces each column to a scalar,
        % giving a 1-by-n row vector.
        C = gb_minbycol (op, A) ;
    elseif (option == 2)
        % C = min (A, [ ], 2) reduces each row to a scalar,
        % giving an m-by-1 column vector.
        C = gb_minbyrow (op, A) ;
    else
        error ('GrB:error', 'invalid option') ;
    end
end

