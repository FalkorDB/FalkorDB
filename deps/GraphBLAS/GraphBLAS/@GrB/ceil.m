function C = ceil (G)
%CEIL round entries of a matrix to nearest integers towards infinity.
%
% See also GrB/floor, GrB/round, GrB/fix.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_ceil (0, G) ;

