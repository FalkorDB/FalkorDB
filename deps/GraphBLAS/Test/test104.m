function test104
%TEST104 export/import

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default')
fprintf ('\ntest104: export/import tests\n') ;

for m = [0 1 5 100]
    for n = [0 1 5 100]
        if (n == 1)
            fmts = [-9:-1 1:9] ;
        else
            fmts = [1:9] ;
        end
        for d = [0 0.1 0.5 inf]
            A = GB_spec_random (m, n, d) ;
            nz = nnz (A.pattern) ;
            is_sparse = (nz < m*n) ;
            fprintf ('.') ;
            for fmat = fmts
                for fexport = 0:14
                    try
                        C1 = GB_mex_export_import (A, fmat, fexport) ;
                        GB_spec_compare (C1, A) ;
                        C2 = GB_mex_export_import (A, fmat, fexport, 1) ;
                        GB_spec_compare (C2, A) ;
                        C3 = GB_mex_export_import (A, fmat, fexport, 502) ;
                        GB_spec_compare (C3, A) ;
                    catch me
                        % should fail if A is sparse and it is attempted to
                        % be exported as full
                        ok = is_sparse && ...
                            (fexport == 6 || fexport == 7 || ...
                             fexport == -6 || fexport == -7 || ...
                             fexport == 16 || fexport == 15 || ...
                             fexport == -16 || fexport == -15) ;
                        if (~ok)
                            % this should not have failed
                            me
                            me.message
                            assert (false) ;
                        end
                    end
                end
            end
        end
    end
end

fprintf ('\ntest104: all tests passed\n') ;
