function f = gb_fmt (A)
%GB_FMT return the format of A as a single string.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[f, s] = GrB.format (A) ;

if (~isempty (s))
    f = [s ' ' f] ;
end

