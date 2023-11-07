function C = eps (G)
%EPS spacing of numbers in a GraphBLAS matrix.
% C = eps (G) returns the spacing of numbers in a floating-point GraphBLAS
% matrix.
%
% See also GrB/isfloat, realmax, realmin.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% FUTURE: GraphBLAS should have a built-in eps unary operator.

% convert to a built-in full matrix and use the built-in eps:

% FUTURE: there should be a sparse version of 'eps'.
% C is full because eps (0) is 2^(-1024).

switch (GrB.type (G))

    case { 'single' }
        C = GrB (eps (single (full (G)))) ;

    case { 'double' }
        C = GrB (eps (double (full (G)))) ;

    case { 'single complex' }
        C = max (eps (single (real (G))), eps (single (imag (G)))) ;

    case { 'double complex' }
        C = max (eps (double (real (G))), eps (double (imag (G)))) ;

    otherwise
        error ('GrB:error', 'input must be floating-point') ;

end

