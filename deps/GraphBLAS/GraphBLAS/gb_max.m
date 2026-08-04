function C = gb_max (ghb, A, B, option)
%GB_MAX implements GrB/max and GhB/max.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (A))
    A = struct (A) ;
end

if (nargin >= 3 && gb_is_grb (B))
    B = struct (B) ;
end

type = gbmex_type (A) ;
if (gb_contains (type, 'complex'))
    error ('GrB:error', 'complex matrices not yet supported') ;
elseif (isequal (type, 'logical'))
    op = '|.logical' ;
else
    op = 'max' ;
end

switch (nargin)
    case 2
        % C = max (A)
        C = gb_max1 (ghb, op, A) ;
    case 3
        % C = max (A,B)
        C = gb_max2 (ghb, op, A, B) ;
    otherwise
        % C = max (A, [ ], option)
        if (~isempty (B))
            error ('GrB:error', ...
                'dimension argument not allowed with 2 input matrices') ;
        end
        C = gb_max3 (ghb, op, A, option) ;
end

