function test156
%TEST156 test assign C=A with typecasting

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

[~, ~, ~, types, ~, ~] = GB_spec_opsall ;
types = types.all ;


for k1 = 1:length(types)
    in_type = types {k1} ;
    A = GB_spec_random (5, 5, 0.5, 10, in_type) ;
    B = GB_spec_random (5, 5, inf, 10, in_type) ;

    for k2 = 1:length(types)
        out_type = types {k2} ;
        C = GB_spec_random (5, 5, 0.5, 10, out_type) ;
        D = GB_spec_random (5, 5, inf, 10, out_type) ;

        % C = A
        C1 = GB_spec_assign (C, [ ], [ ], A, [ ], [ ], [ ], false) ;
        C2 = GB_mex_assign  (C, [ ], [ ], A, [ ], [ ], [ ]) ;
        GB_spec_compare (C1, C2) ;

        % C = B
        C1 = GB_spec_assign (C, [ ], [ ], B, [ ], [ ], [ ], false) ;
        C2 = GB_mex_assign  (C, [ ], [ ], B, [ ], [ ], [ ]) ;
        GB_spec_compare (C1, C2) ;

        % D = A
        C1 = GB_spec_assign (D, [ ], [ ], A, [ ], [ ], [ ], false) ;
        C2 = GB_mex_assign  (D, [ ], [ ], A, [ ], [ ], [ ]) ;
        GB_spec_compare (C1, C2) ;

        % D = B
        C1 = GB_spec_assign (D, [ ], [ ], B, [ ], [ ], [ ], false) ;
        C2 = GB_mex_assign  (D, [ ], [ ], B, [ ], [ ], [ ]) ;
        GB_spec_compare (C1, C2) ;

    end
end
        

fprintf ('test156: all tests passed\n') ;
