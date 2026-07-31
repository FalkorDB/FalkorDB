function s = gb_contains (text, pattern)
%GB_CONTAINS same as contains (text, pattern).  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

s = ~isempty (strfind (text, pattern)) ; %#ok<STREMP>
