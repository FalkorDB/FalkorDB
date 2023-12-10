function test229
%TEST229 set setElement

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('\ntest229: ---- test setElement_* for all types\n') ;

[~,~,~,types,~,~] = GB_spec_opsall ;
types = types.all ;

for k = 1:length (types)
    ctype = types {k} ;
    fprintf ('%s: ', ctype) ;

    for n = [1 10 100]
        for m = [1 10 100]
            for d = [0 .5 inf]
                A = GB_spec_random (m, n, d, 100, ctype) ;

                ntuples = 10 ;
                I = 1 + floor (m * rand (ntuples, 1)) ;
                J = 1 + floor (n * rand (ntuples, 1)) ;
                X = GB_spec_random (ntuples, 1, inf, 100, ctype) ;
                X = full (X.matrix) ;
                I0 = uint64 (I)-1 ;
                J0 = uint64 (J)-1 ;

                C1 = GB_mex_setElement (A, I0, J0, X, true, false) ;
                C2 = GB_mex_setElement (A, I0, J0, X, false, false) ;
                C3 = GB_mex_setElement (A, I0, J0, X, true, true) ;
                C4 = GB_mex_setElement (A, I0, J0, X, true, true) ;

                assert (isequal (C1, C2)) ;
                assert (isequal (C1, C3)) ;
                assert (isequal (C1, C4)) ;

                if (isequal (ctype, 'double'))
                    fprintf ('+') ;
                    C0 = A.matrix ;
                    for k = 1:length(I)
                        C0 (I (k), J (k)) = X (k) ;
                    end
                    assert (isequal (C1.matrix, C0)) ;
                end
            end
        end
    end
    fprintf ('\n') ;
end

fprintf ('\ntest229: all tests passed\n') ;
