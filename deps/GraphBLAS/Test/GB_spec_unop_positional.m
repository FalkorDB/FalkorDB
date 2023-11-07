function z = GB_spec_unop_positional (op, i, j)
%GB_SPEC_UNOP_POSITIONAL compute a unary positional op

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

switch (op)
    case { 'positioni' , 'i'   }
        z = i - 1 ;
    case { 'positioni1', 'i1'  }
        z = i ;
    case { 'positionj' , 'j'   }
        z = j - 1 ;
    case { 'positionj1', 'j1'  }
        z = j ;
    otherwise
        error ('unknown unary positional op') ;
end

z = int64 (z) ;

