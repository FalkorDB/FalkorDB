function s = gb_isfull (A)
%GB_ISFULL determine if all entries are present in a GraphBLAS struct.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[m, n] = gbsize (A) ;
if (isinteger (m))
    % gbsize returms m and n as integer if either m or n are larger
    % than flintmax.  In this case, A must be sparse.
    s = false ;
else
    s = (m*n == gbnvals (A)) ;
end

