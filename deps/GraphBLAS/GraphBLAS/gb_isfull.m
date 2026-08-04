function s = gb_isfull (A)
%GB_ISFULL determine if all entries present in a matrix.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% note that gbmex_nvals requires a wait, but this is required to determine
% if all entries are present anyway.

[m, n] = gb_size (A) ;
s = (m*n == gzb_nvals (A)) ;

