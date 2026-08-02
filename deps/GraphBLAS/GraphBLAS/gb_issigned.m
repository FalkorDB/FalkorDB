function s = gb_issigned (type)
%GB_ISSIGNED Determine if a type is signed or unsigned.  Not user-callable.
% s = gb_issigned (type) returns true if type is the string 'double',
% 'single', 'single complex', 'double complex', 'int8', 'int16', 'int32',
% or 'int64'.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

s = ~ (isequal (type, 'logical') || gb_contains (type, 'uint')) ;

