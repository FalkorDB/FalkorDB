function p = symrcm (G)
%SYMRCM reverse Cuthill-McKee ordering.
% See 'help symrcm' for details.
%
% See also GrB/amd, GrB/colamd, GrB/symamd.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

p = builtin ('symrcm', logical (G)) ;

