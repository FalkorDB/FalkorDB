function C = gb_min (ghb, A, B, option)
%GB_MIN implements GrB/min and GhB/min.  Not user-callable.

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
    op = '&.logical' ;
else
    op = 'min' ;
end

switch (nargin)
    case 2
        % C = min (A)
        C = gb_min1 (ghb, op, A) ;
    case 3
        % C = min (A,B)
        C = gb_min2 (ghb, op, A, B) ;
    otherwise
        % C = min (A, [ ], option)
        if (~isempty (B))
            error ('GrB:error', ...
                'dimension argument not allowed with 2 input matrices') ;
        end
        C = gb_min3 (ghb, op, A, option) ;
end

