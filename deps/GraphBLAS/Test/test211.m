function test211
%TEST211 test iso assign

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

n = 10 ;

% GB_mex_burble (1) ;
Cin.matrix = logical (spones (sprand (n, n, 0.5))) ;
Cin.iso = true ;

op.opname = 'or' ;
op.optype = 'logical' ;

M = sprand (n, n, 0.5) ;
desc = struct ('mask', 'structural') ;

scalar.matrix = true ;
scalar.class = 'logical' ;

for s = [true false]

    scalar.matrix = s ;

    C1 = GB_mex_assign  (Cin, M, [ ], scalar, [ ], [ ], [ ]) ;
    C2 = GB_spec_assign (Cin, M, [ ], scalar, [ ], [ ], [ ], true) ;
    GB_spec_compare (C1, C2) ;
    assert (C1.iso == s) ;

    C1 = GB_mex_assign  (Cin, M, op, scalar, [ ], [ ], [ ]) ;
    C2 = GB_spec_assign (Cin, M, op, scalar, [ ], [ ], [ ], true) ;
    GB_spec_compare (C1, C2) ;
    assert (C1.iso == s) ;

end

GB_mex_burble (0) ;
fprintf ('\ntest211: all tests passed\n') ;

