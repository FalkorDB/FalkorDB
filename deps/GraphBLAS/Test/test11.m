function test11
%TEST11 test GrB_*_extractTuples

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[~, ~, ~, types, ~, ~] = GB_spec_opsall ;
types = types.all ;

fprintf ('\n------------ testing GrB_extractTuples') ;

% type of the output X
for k1 = 1:length (types)
    xtype = types {k1}  ;
    fprintf ('\n%-14s ', xtype) ;

    % type of the matrix A
    for k2 = 1:length (types)
        atype = types {k2}  ;

        % create a matrix
        fprintf ('.') ;
        for m = [1 10 25]
            for n = [1 10 25]
                clear A
                A = GB_spec_random (m, n, 0.1, 32, atype) ;

                clear B
                B = GB_spec_random (m*n, 1, 0.1, 32, atype) ;

                for A_is_hyper = 0:1
                for A_is_csc   = 0:1
                A.is_hyper = A_is_hyper ;
                A.is_csc   = A_is_csc   ;

                [I1, J1, X1] = GB_mex_extractTuples  (A, xtype) ;
                [I2, J2, X2] = GB_spec_extractTuples (A, xtype) ;

                % If A is CSR, the extraction returns tuples in row major
                % order, but GB_spec_extractTuples always returns the tuples in
                % column major order.  Either way is fine since the order does
                % not matter.

                [~,p1] = sortrows ([I1 J1]) ;
                I1 = I1 (p1) ; 
                J1 = J1 (p1) ; 
                X1 = X1 (p1) ; 

                [~,p2] = sortrows ([I2 J2]) ;
                I2 = I2 (p2) ; 
                J2 = J2 (p2) ; 
                X2 = X2 (p2) ; 

                assert (isequal (I1, I2)) ;
                assert (isequal (J1, J2)) ;
                assert (isequal (X1, X2)) ;

                end
                end

                [I1, J1, X1] = GB_mex_extractTuples  (B, xtype) ;
                [I2, J2, X2] = GB_spec_extractTuples (B, xtype) ;

                assert (isequal (I1, I2)) ;
                assert (isequal (J1, J2)) ;
                assert (isequal (X1, X2)) ;

            end
        end
    end
end

fprintf ('\ntest11: all tests passed\n') ;

