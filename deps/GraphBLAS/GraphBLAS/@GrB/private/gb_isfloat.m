function s = gb_isfloat (type)
%GB_ISFLOAT true for floating-point GraphBLAS types.
% Implements s = isfloat (type (G))

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

s = gb_contains (type, 'double') || gb_contains (type, 'single') ;

