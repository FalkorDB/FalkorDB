function test105
%TEST105 eWiseAdd with hypersparse matrices

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng 'default'
fprintf ('\ntest105: eWiseAdd with hypersparse\n') ;

for m = [0 1 5 100]
    for n = [0 1 5 100]
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

                    % C = A+B, no mask
                    C0 = GB_spec_Matrix_eWiseAdd (C, [ ], [ ], ...
                                                 'plus', A, B, [ ], 'test') ;
                    C1 = GB_mex_Matrix_eWiseAdd  (C, [ ], [ ], ...
                                                 'plus', A, B, [ ], 'test') ;
                    GB_spec_compare (C0, C1) ;

                    % C = A+B, with mask
                    C0 = GB_spec_Matrix_eWiseAdd (C, M, [ ], ...
                                                 'plus', A, B, [ ], 'test') ;
                    C1 = GB_mex_Matrix_eWiseAdd  (C, M, [ ], ...
                                                 'plus', A, B, [ ], 'test') ;
                    GB_spec_compare (C0, C1) ;

                end
            end
        end
    end
end

fprintf ('\ntest105: all tests passed\n') ;
