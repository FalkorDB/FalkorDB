function C = full (A, type, identity)
%FULL convert a matrix into a GraphBLAS full matrix.
% C = full (A, type, identity) converts the matrix A into a GraphBLAS full
% matrix C of the given type, by inserting identity values.  The type may
% be any GraphBLAS type: 'double', 'single', 'single complex', 'double
% complex', 'logical', 'int8', 'int16', 'int32', 'int64', 'uint8',
% 'uint16', 'uint32', or 'uint64'.
%
% If not present, the type defaults to the same type as A, and the
% identity defaults to zero.  A may be any matrix (GraphBLAS or built-in)
% To use this method for a built-in matrix A, use a GraphBLAS identity
% value such as GrB(0), or use C = full (GrB (A)).  Note that issparse (C)
% is true, since issparse (A) is true for any GraphBLAS matrix A.
%
% Examples:
%
%   G = GrB (sprand (5, 5, 0.5))        % GraphBLAS sparse matrix
%   C = full (G)                        % add explicit zeros
%   C = full (G, 'double', inf)         % add explicit inf's
%
%   A = speye (2)
%   C = full (GrB (A), 'double', 0)      % full GrB matrix C, from A
%   C = full (GrB (A))                   % same matrix C
%
% See also GrB/issparse, sparse, cast, GrB.type, GrB, GrB.isfull.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

A_is_GrB = isobject (A) ;
if (A_is_GrB)
    % A is a GraphBLAS matrix
    Q = A.opaque ;
else
    % A is a built-in matrix
    Q = A ;
end

if (nargin < 2)
    type = gbtype (Q) ;
    right_type = true ;
else
    right_type = isequal (type, gbtype (Q)) ;
end

if (gb_isfull (Q) && right_type)

    % nothing to do, A is already full and has the right type
    if (A_is_GrB)
        % A is already a GrB matrix, return it as-is
        C = A ;
    else
        % convert A into a GrB matrix
        C = GrB (A) ;
    end

else

    % convert A to a full GraphBLAS matrix
    if (nargin < 3)
        identity = 0 ;
    elseif (isobject (identity))
        identity = identity.opaque ;
    end
    C = GrB (gbfull (Q, type, identity)) ;

end

