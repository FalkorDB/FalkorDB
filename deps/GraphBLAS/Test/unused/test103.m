function test103
%TEST103 test aliases in GrB_transpose

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng 'default'
fprintf ('\ntest103: test aliases in GrB_transpose\n') ;

for m = [1 5 10]
    for n = [1 5 10]
        for d = [0 0.1 .5 1]
            for is_csc = 0:1
                for is_hyper = 0:1

                    if (is_hyper)
                        hyper_switch = 1 ;
                    else
                        hyper_switch = 0 ;
                    end

                    % A = GB_spec_random (m, n, d, scale, class,
                    % is_csc,is_hyper,hyper_switch)

                    C = GB_spec_random (m, n, d, 100, 'double', ...
                        is_csc, is_hyper, hyper_switch) ;

                    M = sparse (ones (m, n)) ;

                    A = GB_spec_random (m, n, d, 100, 'double', ...
                        is_csc, is_hyper, hyper_switch) ;

                    % C<M>=A, to test shallow cast
                    desc.inp0 = 'tran' ;
                    C2a = GB_spec_transpose (C, M, 'plus', A, desc) ;
                    C2b = GB_mex_transpose  (C, M, 'plus', A, desc, 'test') ;
                    GB_spec_compare (C2a, C2b) ;

                    C2a = GB_spec_transpose (C, M, [ ], A, desc) ;
                    C2b = GB_mex_transpose  (C, M, [ ], A, desc, 'test') ;
                    GB_spec_compare (C2a, C2b) ;

                    C2a = GB_spec_transpose (C, [ ], [ ], A, desc) ;
                    C2b = GB_mex_transpose  (C, [ ], [ ], A, desc, 'test') ;
                    GB_spec_compare (C2a, C2b) ;

                    C3a = GB_spec_transpose (C,  C,  'plus', A, desc) ;
                    C3b = GB_mex_transpose  (C,  C,  'plus', A, desc, 'test') ;
                    C3c = GB_mex_transpose  (C, 'C', 'plus', A, desc, 'test') ;
                    GB_spec_compare (C3a, C3b) ;
                    GB_spec_compare (C3a, C3c) ;

                    C3a = GB_spec_transpose (C,  C,  [ ], A, desc) ;
                    C3b = GB_mex_transpose  (C,  C,  [ ], A, desc, 'test') ;
                    C3c = GB_mex_transpose  (C, 'C', [ ], A, desc, 'test') ;
                    GB_spec_compare (C3a, C3b) ;
                    GB_spec_compare (C3a, C3c) ;

                    C4a = GB_spec_transpose (C,  C,  'plus',  C,  desc) ;
                    C4b = GB_mex_transpose  (C,  C,  'plus',  C,  desc, 'test');
                    GB_spec_compare (C4a, C4b) ;
                    C4c = GB_mex_transpose  (C, 'C', 'plus', 'C', desc, 'test');
                    GB_spec_compare (C4a, C4c) ;

                    C4a = GB_spec_transpose (C,  C,  [ ],  C,  desc) ;
                    C4b = GB_mex_transpose  (C,  C,  [ ],  C,  desc, 'test');
                    GB_spec_compare (C4a, C4b) ;
                    C4c = GB_mex_transpose  (C, 'C', [ ], 'C', desc, 'test');
                    GB_spec_compare (C4a, C4c) ;

                end
            end
        end
    end
end

fprintf ('\ntest103: all tests passed\n') ;

