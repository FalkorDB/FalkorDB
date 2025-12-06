function test301
%TEST301 test GrB_assign with aliased inputs, C<C,struct>(:,:) += A 

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('test301 ------------------  assign alias tests (method 27)\n') ;

rng ('default') ;

desc.mask = 'structural' ;

seed = 1 ;
for m = [1 5 10 200]
    for n = [1 5 10 200]
        fprintf ('.') ;
        for t1 = [10 10*n n^2]
            for t2 = [10 10*n n^2]
                for trial = 1:30
                    A = GB_mex_random (m, n, t1, 0, seed) ; seed = seed + 1 ;
                    C = GB_mex_random (m, n, t2, 0, seed) ; seed = seed + 1 ;
                    if (mod (trial, 4) == 0)
                        C (:,end) = 1 ;
                    elseif (mod (trial, 4) == 1)
                        C (1:2:end,end) = 1 ;
                    end
                    M = C ;

                    % C<C,struct> += A
                    C1 = GB_mex_assign_method27 (C, 'plus', A) ;
                    C2 = GB_mex_assign (C, M, 'plus', A, [ ], [ ], desc, 0) ;
                    if (~isequal (C1, C2))
                        C1
                        C2
                    end
                    assert (isequal (C1, C2)) ;

                    % A<A,struct> += C
                    M = A ;
                    C1 = GB_mex_assign_method27 (A, 'plus', C) ;
                    C2 = GB_mex_assign (A, M, 'plus', C, [ ], [ ], desc, 0) ;
                    assert (isequal (C1, C2)) ;
                end
            end
        end
    end
end

fprintf ('\ntest301: assign alias tests passed\n') ;

