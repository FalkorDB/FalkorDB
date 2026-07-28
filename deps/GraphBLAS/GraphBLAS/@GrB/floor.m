function C = floor (G)
%FLOOR round entries to nearest integers towards -infinity.
% C = floor (G) rounds the entries in the matrix G to the nearest integers
% towards -infinity.
%
% See also GrB/ceil, GrB/round, GrB/fix.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_floor (0, G) ;

