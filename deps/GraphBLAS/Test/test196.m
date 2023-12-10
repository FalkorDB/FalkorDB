function test196
%TEST196 test large hypersparse concat

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;
m = 2e6 ;
n = 1e6 ;
d = 100 / n^2 ;
A1.matrix = sprand (m, n, d) ;      A1.sparsity = 1 ;
A2.matrix = sprand (m, 2*n, d) ;    A2.sparsity = 1 ;
A3.matrix = sprand (2*m, n, d) ;    A3.sparsity = 1 ;
A4.matrix = sprand (2*m, 2*n, d) ;  A4.sparsity = 1 ;

for is_csc = 0:1
    A1.is_csc = is_csc ;
    A2.is_csc = is_csc ;
    A3.is_csc = is_csc ;
    A4.is_csc = is_csc ;
    Tiles = cell (2,2) ;
    Tiles {1,1} = A1 ;
    Tiles {1,2} = A2 ;
    Tiles {2,1} = A3 ;
    Tiles {2,2} = A4 ;
    C1 = GB_mex_concat (Tiles, 'double', 1) ;
    C2 = [ A1.matrix A2.matrix ; A3.matrix A4.matrix ] ;
    assert (isequal (C1.matrix, C2)) ;
end

% convert only some to iso
A1.matrix = spones (A1.matrix) ; A1.iso = 1 ;
A2.iso = 0 ;
A3.iso = 0 ;
A4.iso = 0 ;

for is_csc = 0:1
    A1.is_csc = is_csc ;
    A2.is_csc = is_csc ;
    A3.is_csc = is_csc ;
    A4.is_csc = is_csc ;
    Tiles = cell (2,2) ;
    Tiles {1,1} = A1 ;
    Tiles {1,2} = A2 ;
    Tiles {2,1} = A3 ;
    Tiles {2,2} = A4 ;
    for C_is_csc = 0:1
        C1 = GB_mex_concat (Tiles, 'double', C_is_csc) ;
        C2 = [ A1.matrix A2.matrix ; A3.matrix A4.matrix ] ;
        assert (isequal (C1.matrix, C2)) ;
    end
end

% convert all to iso
A1.matrix = spones (A1.matrix) ; A1.iso = 1 ;
A2.matrix = spones (A1.matrix) ; A2.iso = 1 ;
A3.matrix = spones (A1.matrix) ; A3.iso = 1 ;
A4.matrix = spones (A1.matrix) ; A4.iso = 1 ;

for is_csc = 0:1
    A1.is_csc = is_csc ;
    A2.is_csc = is_csc ;
    A3.is_csc = is_csc ;
    A4.is_csc = is_csc ;
    Tiles = cell (2,2) ;
    Tiles {1,1} = A1 ;
    Tiles {1,2} = A2 ;
    Tiles {2,1} = A3 ;
    Tiles {2,2} = A4 ;
    C1 = GB_mex_concat (Tiles, 'double', 1) ;
    C2 = [ A1.matrix A2.matrix ; A3.matrix A4.matrix ] ;
    assert (isequal (C1.matrix, C2)) ;
end

fprintf ('test196: all tests passed\n') ;

