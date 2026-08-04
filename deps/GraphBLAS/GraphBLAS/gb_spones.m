function C = gb_spones (ghb, G, type)
%GB_SPONES return pattern of GraphBLAS matrix.  Not user-callable.
% Implements C = spones (G).

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (G))
    G = struct (G) ;
end

if (nargin < 3)
    switch (gbmex_type (G))
        case { 'single complex' }
            op = '1.single' ;
        case { 'double complex' }
            op = '1.double' ;
        otherwise
            op = '1' ;
    end
else
    if (~ischar (type))
        error ('GrB:error', 'type must be a string') ;
    end
    op = ['1.' type] ;
end

C = gzb_apply (ghb, op, G) ;

