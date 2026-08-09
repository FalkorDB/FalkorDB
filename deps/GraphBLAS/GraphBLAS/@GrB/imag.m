function C = imag (G)
%IMAG complex imaginary part.
% C = imag (G) returns the imaginary part of G.
%
% See also GrB/conj, GrB/real.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_imag (0, G) ;

