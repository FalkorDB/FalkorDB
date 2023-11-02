function s = isbycol (A)
%GRB.ISBYCOL true if A is stored by column, false if by column.
% s = GrB.isbycol (A) is true if A is stored by column, false if by row.
% A may be a GraphBLAS matrix or built-in matrix (sparse or full).
% Built-in matrices are always stored by column.
%
% See also GrB.isbyrow, GrB.format.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (isobject (A))
    A = A.opaque ;
    s = isequal (gbformat (A), 'by col')  ;
else
    s = true ;
end

