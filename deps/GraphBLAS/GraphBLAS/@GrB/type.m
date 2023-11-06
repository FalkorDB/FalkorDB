function s = type (X)
%GRB.TYPE get the type of a built-in or GraphBLAS matrix.
% s = GrB.type (X) returns the type of a GraphBLAS matrix X as a string:
% 'logical', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16',
% 'uint32', 'uint64', 'single', 'double', 'single complex', and 'double
% complex'.  Note that for GraphBLAS matrices, 'complex' is treated as a
% type, not an attribute, which differs from the built-in convention.
%
% For a GraphBLAS matrix X of any type, class (X) returns 'GrB'.
% GraphBLAS does not overload the built-in 'class' or 'isobject' functions,
% but it does overload the 'isa' function, which queries the type of a
% GraphBLAS matrix.
%
% If X is not a GraphBLAS matrix, GrB.type (X) is the same as class (X),
% except when X is a built-in single complex or double complex matrix, in
% which case GrB.type (X) is 'single complex' or 'double complex',
% respectively.  The built-in class (X) is 'single' if X is single complex
% and 'double' if X is double complex.
%
% Examples:
%
%   A = int8 (magic (4))
%   G = GrB (A)
%   class (A)
%   GrB.type (A)
%   class (G)
%   GrB.type (G)
%   isa (G, 'int8')
%   isa (A, 'int8')
%
%   A = single (pi + 1i)
%   G = GrB (A)
%   class (A)
%   GrB.type (A)
%   class (G)
%   GrB.type (G)
%   isa (G, 'single complex')
%   isa (A, 'single complex')
%
% See also class, isa, GrB/isa, GrB.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (isobject (X))
    X = X.opaque ;
end

s = gbtype (X) ;

