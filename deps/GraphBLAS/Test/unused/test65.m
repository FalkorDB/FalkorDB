function test65
%TEST65 test type casting

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

X = logical (rand (4) > 0.5) ;
C = GB_mex_cast (X, 'logical', 1) ;
assert (isequal (X,C)) ;

    A = 100 * randn (4) ;

    X = int8 (A) ;
    C = GB_mex_cast (X, 'int8', 1) ;
    assert (isequal (X,C)) ;

    X = uint8 (A) ;
    C = GB_mex_cast (X, 'uint8', 1) ;
    assert (isequal (X,C)) ;

    X = int16 (A) ;
    C = GB_mex_cast (X, 'int16', 1) ;
    assert (isequal (X,C)) ;

    X = uint16 (A) ;
    C = GB_mex_cast (X, 'uint16', 1) ;
    assert (isequal (X,C)) ;

    X = int32 (A) ;
    C = GB_mex_cast (X, 'int32', 1) ;
    assert (isequal (X,C)) ;

    X = uint32 (A) ;
    C = GB_mex_cast (X, 'uint32', 1) ;
    assert (isequal (X,C)) ;

    X = int32 (A) ;
    C = GB_mex_cast (X, 'int32', 1) ;
    assert (isequal (X,C)) ;

    X = uint32 (A) ;
    C = GB_mex_cast (X, 'uint32', 1) ;
    assert (isequal (X,C)) ;

    X = single (A) ;
    C = GB_mex_cast (X, 'single', 1) ;
    assert (isequal (X,C)) ;

    X = double (A) ;
    C = GB_mex_cast (X, 'double', 1) ;
    assert (isequal (X,C)) ;

    fprintf ('\ntest65: all tests passed\n') ;

