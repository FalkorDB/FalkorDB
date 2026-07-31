function C = gb_tri (ghb, op, G, k)
%GB_TRI implements tril and triu for GrB and GhB.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

k = gb_get_scalar (k) ;
C = gzb_select (ghb, op, G, k) ;

