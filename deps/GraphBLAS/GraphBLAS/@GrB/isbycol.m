function s = isbycol (A)
%GRB.ISBYCOL true if A is stored by column, false if by column.
% s = GrB.isbycol (A) is true if A is stored by column, false if by row.  A may
% be a GraphBLAS matrix or built-in matrix (sparse or full).  Built-in matrices
% are always stored by column.
%
% See also GrB.isbyrow, GrB.format, GhB.set, GhB.get.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (isobject (A))
    if (gb_is_grb (A))
        A = struct (A) ;
    end
    s = isequal (gbmex_format (A), 'by col')  ;
else
    s = true ;
end

