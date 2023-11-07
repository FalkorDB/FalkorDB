function C = gb_spones (G, type)
%GB_SPONES return pattern of GraphBLAS matrix.
% Implements C = spones (G).

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 1)
    switch (gbtype (G))
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

C = gbapply (op, G) ;

