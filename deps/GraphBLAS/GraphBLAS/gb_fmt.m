function f = gb_fmt (A)
%GB_FMT return the format of A as a single string.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[f, s] = gbmex_format (A) ;

if (~isempty (s))
    f = [s ' ' f] ;
end

