function gbtest105 (ghb)
%GBTEST105 test logical assignment with iso matrices

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

types = gbtest_types ;

X = magic (5) ;
Y = 10 * rand (5) ;
M = logical (rand (5) > 0.5) ;

for k = 1:length (types)
    type = types {k} ;

    if (isequal (type, 'logical'))
        A = (mod (X,2) == 0) ;
        C = (Y > 5) ;
    elseif (isequal (type, 'single complex'))
        A = cast (X, 'single') + 1i * rand (5, 'single') ;
        C = cast (Y, 'single') + 1i * rand (5, 'single') ;
    elseif (isequal (type, 'double complex'))
        A = cast (X, 'double') + 1i * rand (5) ;
        C = cast (Y, 'double') + 1i * rand (5) ;
    else
        A = cast (X, type) ;
        C = cast (Y, type) ;
    end

    % pure MATLAB
    C1 = C ;
    C1 (M) = A (M) ;

    % pure GraphBLAS
    G1 = gtb (ghb, C) ;
    H = gtb (ghb, A) ;
    G1 (M) = H (M) ;

    assert (isequal (G1, C1)) ;

    % now make the A and H matrices iso
    if (gb_contains (type, 'complex'))
        A (A ~= 0) = 1i ;
        H (H ~= 0) = 1i ; 
        H = gtb_prune (ghb, H) ;
        H = spones (H) ;
        H = H * 1i ;
        H = gtb (ghb, H, type) ;
    else
        A (A ~= 0) = 1 ;
        H (H ~= 0) = 1 ; 
        H = gtb_prune (ghb, H) ;
        H = spones (H) ;
    end
    assert (isequal (full (H), A)) ;


    % pure MATLAB
    C1 = C ;
    C1 (M) = A (M) ;

    % pure GraphBLAS
    G1 = gtb (ghb, C) ;
    G1 (M) = H (M) ;
    G1 = gtb_prune (ghb, G1) ;

    assert (isequal (full (G1), C1)) ;
end

fprintf ('\ngbtest105 (%d: all tests passed\n', ghb) ;

