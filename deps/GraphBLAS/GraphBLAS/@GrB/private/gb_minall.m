function C = gb_minall (op, A)
%GB_MINALL reduce a matrix to a scalar
% Implements C = min (A, [ ], 'all') ;

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gbreduce (op, A) ;
[m, n] = gbsize (A) ;
if ((m*n ~= gbnvals (A)) && gb_scalar (C) >= 0)
    % A is not full, and the min of the entries present is >= 0,
    % so C is an empty scalar (an implicit zero)
    C = gbnew (1, 1, gbtype (C)) ;
end

