function test137
%TEST137 GrB_eWiseMult with FIRST and SECOND operators

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('test137: GrB_eWiseMult with FIRST and SECOND operators\n') ;

rng ('default') ;

n = 100 ;
for k = [false true]
    fprintf ('builtin_complex: %d\n', k) ;
    GB_builtin_complex_set (k) ;

    A = sprand (n, n, 0.1) ;
    Z = sprand (n, n, 0.1) + 1i * sprand (n, n, 0.1) ;
    S = spones (Z) ;
    C = sparse (n, n) ;

    try
        % this is an error
        C1 = GB_mex_Matrix_eWiseMult (C, [ ], [ ], 'times', A, Z, [ ]) ;
        assert (false) ;
    catch
        assert (true) ;
    end

    C0 = A .* S ;
    C1 = GB_mex_eWiseMult_first (C, [ ], [ ], [ ], A, Z, [ ]) ;
    assert (isequal (C0, C1.matrix)) ;

    C0 = S .* A ;
    C1 = GB_mex_eWiseMult_second (C, [ ], [ ], [ ], Z, A, [ ]) ;
    assert (isequal (C0, C1.matrix)) ;
end


GB_builtin_complex_set (true) ;
fprintf ('test137: all tests passed\n') ;
