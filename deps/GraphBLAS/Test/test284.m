function test284
%TEST284 test GrB_mxm using indexop-based semirings

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('\ntest284: GrB_mxm with indexop-based semirings\n') ;

rng ('default') ;

n = 10 ;
A = GB_spec_random (n, n, 0.3, 100, 'double') ;
B = GB_spec_random (n, n, 0.3, 100, 'double') ;
D = speye (n) ;

% all variations:
adds = {'min', 'max', 'plus', 'times' } ;
mults = {'firsti', 'firsti1', 'firstj', 'firstj1', ...
         'secondi', 'secondi1', 'secondj', 'secondj1' } ;

% just a few:
adds = {'min', 'plus' } ;
mults = {'firsti1', 'secondi1', 'secondj' } ;

for A_is_csc = 0:1
    A.is_csc = A_is_csc ;

    for B_is_csc = 0:1
        B.is_csc = A_is_csc ;

        for C_is_csc = 0:1

            for A_sparsity = [1 2 4]
                if (A_sparsity == 0)
                    A.is_hyper = 0 ;
                    A.is_bitmap = 0 ;
                    A.sparsity = 2 ;    % sparse
                elseif (A_sparsity == 1)
                    A.is_hyper = 1 ;
                    A.is_bitmap = 0 ;
                    A.sparsity = 1 ;    % hypersparse
                else
                    A.is_hyper = 0 ;
                    A.is_bitmap = 1 ;
                    A.sparsity = 4 ;    % bitmap
                end

                for B_sparsity = [1 2 4]
                    if (B_sparsity == 0)
                        B.is_hyper = 0 ;
                        B.is_bitmap = 0 ;
                        B.sparsity = 2 ;    % sparse
                    elseif (B_sparsity == 1)
                        B.is_hyper = 1 ;
                        B.is_bitmap = 0 ;
                        B.sparsity = 1 ;    % hypersparse
                    else
                        B.is_hyper = 0 ;
                        B.is_bitmap = 1 ;
                        B.sparsity = 4 ;    % bitmap
                    end

                    for at = 0:1
                        for bt = 0:1
                            for method = [0 7081 7083 7084 7085]
                                % C = A*B, A'*B, A*B', or A'*B'
                                for k1 = 1:length (adds)
                                    add = adds {k1} ;
                                    for k2 = 1:length (mults)
                                        mult = mults {k2} ;
%                                       fprintf ('\n(%s,%s,%d,%d):\n', ...
%                                           add, mult, at, bt) ;
                                        C1 = GB_mex_AxB_idx (A, B, at, bt, ...
                                            method, C_is_csc, 1, add, mult) ;
                                        C2 = GB_mex_AxB_idx (A, B, at, bt, ...
                                            method, C_is_csc, 0, add, mult) ;
                                        GB_spec_compare (C1, C2) ;
                                    end
                                end
                            end
                            fprintf ('.') ;
                        end
                    end

                    for at = 0:1
                        % C = A*D, A'*D
                        for k1 = 1:length (adds)
                            add = adds {k1} ;
                            for k2 = 1:length (mults)
                                mult = mults {k2} ;
%                               fprintf ('\n(%s,%s,%d): D*B\n', ...
%                                   add, mult, at) ;
                                C1 = GB_mex_AxB_idx (A, D, at, 0, ...
                                    0, C_is_csc, 1, add, mult) ;
                                C2 = GB_mex_AxB_idx (A, D, at, 0, ...
                                    0, C_is_csc, 0, add, mult) ;
                                GB_spec_compare (C1, C2) ;
                            end
                        end
                    end
                    fprintf ('.') ;

                    for bt = 0:1
                        % C = D*B, D*B'
                        for k1 = 1:length (adds)
                            add = adds {k1} ;
                            for k2 = 1:length (mults)
                                mult = mults {k2} ;
%                               fprintf ('\n(%s,%s,%d): D*B\n', ...
%                                   add, mult, bt) ;
                                C1 = GB_mex_AxB_idx (D, B, 0, bt, ...
                                    0, C_is_csc, 1, add, mult) ;
                                C2 = GB_mex_AxB_idx (D, B, 0, bt, ...
                                    0, C_is_csc, 0, add, mult) ;
                                GB_spec_compare (C1, C2) ;
%                               fprintf ('.') ;
                            end
                        end
                    end
                    fprintf ('.') ;

                end
            end
        end
    end
end

fprintf ('\ntest284: all tests passed\n') ;

