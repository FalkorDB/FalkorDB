function test106
%TEST106 GxB_subassign with alias

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng 'default'
fprintf ('\ntest106: GxB_subassign with alias\n') ;

for m = [0 1 5 100]
    for n = [0 1 5 100]
        I1 = randperm (m) ;
        J1 = randperm (n) ;
        I0 = uint64 (I1) - 1 ;
        J0 = uint64 (J1) - 1 ;
        for d = [0 0.1 0.5 1]
            for is_csc = 0:1
                for is_hyper = 0:1

                    if (is_hyper)
                        hyper_switch = 1 ;
                    else
                        hyper_switch = 0 ;
                    end

                    A = GB_spec_random (m, n, d, 100, 'double', ...
                        is_csc, is_hyper, hyper_switch) ;
                    B = GB_spec_random (m, n, d, 100, 'double', ...
                        is_csc, is_hyper, hyper_switch) ;
                    C = GB_spec_random (m, n, d, 100, 'double', ...
                        is_csc, is_hyper, hyper_switch) ;
                    M = GB_spec_random (m, n, d, 100, 'double', ...
                        is_csc, is_hyper, hyper_switch) ;

                    C1a = GB_mex_subassign  (C, [ ], [ ],  C,  I0, J0, [ ]) ;
                    C2  = GB_spec_subassign (C, [ ], [ ],  C,  I1, J1, [ ], 0) ;
                    GB_spec_compare (C1a, C2) ;
                    C1b = GB_mex_subassign  (C, [ ], [ ], 'C', I0, J0, [ ]) ;
                    GB_spec_compare (C1b, C2) ;

                    C1a = GB_mex_subassign  (C,  C,  [ ], A, I0, J0, [ ]) ;
                    C2  = GB_spec_subassign (C,  C,  [ ], A, I1, J1, [ ], 0) ;
                    GB_spec_compare (C1a, C2) ;
                    C1b = GB_mex_subassign  (C, 'C', [ ], A, I0, J0, [ ]) ;
                    GB_spec_compare (C1b, C2) ;

                end
            end
        end
    end
end

fprintf ('\ntest106: all tests passed\n') ;

