function z = GB_spec_binop_positional (op, ia, ja, ib, jb)
%GB_SPEC_BINOP_POSITIONAL compute a binary positional op

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

switch (op)
    case { 'firsti'   , '1sti'  }
        z = ia - 1 ;
    case { 'firsti1'  , '1sti1' }
        z = ia ;
    case { 'firstj'   , '1stj'  }
        z = ja - 1 ;
    case { 'firstj1'  , '1stj1' }
        z = ja ;
    case { 'secondi'  , '2ndi'  }
        z = ib - 1 ;
    case { 'secondi1' , '2ndi1' }
        z = ib ;
    case { 'secondj'  , '2ndj'  }
        z = jb - 1 ;
    case { 'secondj1' , '2ndj1' }
        z = jb ;
    otherwise
        error ('unknown binary positional op') ;
end

z = int64 (z) ;

