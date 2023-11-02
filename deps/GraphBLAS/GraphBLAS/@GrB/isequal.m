function s = isequal (A, B)
%ISEQUAL True if matrices are equal and have the same type.
% s = isequal (A,B) is true of A and B have the same size, type, pattern,
% and values.  If A and B have different types, this function returns
% false.  Typecast A and B to a common type to compare their values
% (which is the behavior of the built-in isequal(A,B) function).
%
% If A is a GraphBLAS matrix with an explicit entry equal to zero, but in
% B that entry is not present, then isequal (A,B) returns false.  To drop
% them, use isequal (GrB.prune(A), GrB.prune(B)).
%
% The input matrices may be either GraphBLAS and/or built-in matrices, in
% any combination.  A and B do not have to be the same class.  For
% example, isequal (A, GrB (A)) is always true.
%
% See also isequal, GrB/eq, isequaln.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (isobject (A))
    A = A.opaque ;
end
if (isobject (B))
    B = B.opaque ;
end

s = gbisequal (A, B) ;

