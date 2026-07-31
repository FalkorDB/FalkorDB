function C = extract (arg1, arg2, arg3, arg4, arg5, arg6, arg7)
%GHB.EXTRACT extract sparse submatrix.
%
% syntax for a new matrix C:                        computation:
% C = GhB.extract (A, I, J, desc)                   % C = A(I,J)
% C = GhB.extract (Cin, accum, A, I, J, desc)       % C = Cin ; C += A(I,J)
% C = GhB.extract (Cin, M, A, I, J, desc)           % C = Cin ; C<M> = A(I,J)
% C = GhB.extract (Cin, M, accum, A, I, J, desc)    % C = Cin ; C<M> += A(I,J)
%
% in-place syntax:
% GhB.extract (C, A, I, J, desc)                    % C = A(I,J)
% GhB.extract (C, accum, A, I, J, desc)             % C += A(I,J)
% GhB.extract (C, M, A, I, J, desc)                 % C<M> = A(I,J)
% GhB.extract (C, M, accum, A, I, J, desc)          % C<M> += A(I,J)
%
% A is a required parameter.  All others are optional.  The arguments are
% parsed according to their type.  Arguments with different types can appear in
% any order:
%
%   Cin, M, A:  2 or 3 GraphBLAS/built-in sparse/full matrices.
%               The first three matrix inputs are Cin, M, and A.
%               If 2 matrix inputs are present, they are Cin (or C) and A.
%   accum:      an optional string
%   I,J:        cell arrays:  with no cell inputs: I = { } and J = { }.  with
%               one cell input, I is present and J = { }.  with two cell
%               inputs, I is the first cell input and J is the 2nd cell input.
%   desc:       an optional struct; must appear as the last argument
%
% desc: see 'help GrB.descriptorinfo' for details.
%
% I and J are cell arrays.  I contains 0, 1, 2, or 3 items:
%
%   0:  { }     This is ':', like A(:,J), refering to all m
%               rows, if A is m-by-n.
%
%   1:  { I }   1D list of row indices, like A(I,J).
%
%   2:  { start,fini }  start and fini are scalars, defining I = start:fini.
%
%   3:  { start,inc,fini } start, inc, and fini are scalars,
%               defining I = start:inc:fini.
%
% The J argument is identical, except that it is a list of column indices of A.
% If only one cell array is provided, J = {  } is implied, refering to all n
% columns of A, like A(I,:).  GhB.extract does not support linear indexing of a
% 2D matrix, as in C=A(I) when A is a 2D matrix.
%
% If neither I nor J are provided on input, then this implies both I = { } and
% J = { }, or A(:,:), refering to all rows and columns of A.
%
% desc.base modifies how I, start, and fini are interpretted.  If desc.base is
% 'zero-based' then they are interpretted as zero-based indices, where 0 is the
% first row or column.  If desc.base is 'one-based' (which is the default),
% then indices are intrepetted as 1-based.
%
% accum: an optional binary operator, defined by a string ('+.double') for
%   example.  This allows for C(I,J) = C(I,J) + A to be computed.  If not
%   present, no accumulator is used and C(I,J)=A is computed.  In the
%   computations listed above it is shown as "+=" but any binary operator may
%   be used.  See 'help GrB.binopinfo' for available binary operators.
%
% M: an optional mask matrix, the same size as C.
%
% C or Cin: an optional input matrix, containing the initial content of the
% matrix C.  For the in-place syntax, the GhB matrix C is modified in-place.
% If present, the C or Cin argument has size length(I)-by-length(J).
%
% Except for C for the inplace syntax, all input matrices may be either
% GraphBLAS or built-in matrices, in any combination.  C is returned as a
% GraphBLAS GhB matrix.
%
% Example:
%
%   A = sprand (5, 4, 0.5)
%   I = [2 1 5]
%   J = [3 3 1 2]
%   C = GhB.extract (A, {I}, {J})
%   C2 = A (I,J)
%   C2 - C
%
% See also GhB/subsref, GrB.binopinfo.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargout == 0)
    narginchk (2, 7) ;
else
    narginchk (1, 7) ;
end

if (gb_is_grb (arg1))
    arg1 = struct (arg1) ;
end

if (nargin >= 2 && gb_is_grb (arg2))
    arg2 = struct (arg2) ;
end

if (nargin >= 3 && gb_is_grb (arg3))
    arg3 = struct (arg3) ;
end

if (nargin >= 4 && gb_is_grb (arg4))
    arg4 = struct (arg4) ;
end

if (nargin >= 5 && gb_is_grb (arg5))
    arg5 = struct (arg5) ;
end

if (nargin >= 6 && gb_is_grb (arg6))
    arg6 = struct (arg6) ;
end

% arg7: if present, it must be the descriptor

if (nargout == 0)
    switch (nargin)
        case 2
            gbmex_extract (1, arg1, arg2) ;
        case 3
            gbmex_extract (1, arg1, arg2, arg3) ;
        case 4
            gbmex_extract (1, arg1, arg2, arg3, arg4) ;
        case 5
            gbmex_extract (1, arg1, arg2, arg3, arg4, arg5) ;
        case 6
            gbmex_extract (1, arg1, arg2, arg3, arg4, arg5, arg6) ;
        case 7
            gbmex_extract (1, arg1, arg2, arg3, arg4, arg5, arg6, arg7) ;
    end
else
    switch (nargin)
        case 1
            [C_opaque, kind] = gbmex_extract (1, arg1) ;
        case 2
            [C_opaque, kind] = gbmex_extract (1, arg1, arg2) ;
        case 3
            [C_opaque, kind] = gbmex_extract (1, arg1, arg2, arg3) ;
        case 4
            [C_opaque, kind] = gbmex_extract (1, arg1, arg2, arg3, arg4) ;
        case 5
            [C_opaque, kind] = gbmex_extract (1, arg1, arg2, arg3, arg4, ...
                arg5) ;
        case 6
            [C_opaque, kind] = gbmex_extract (1, arg1, arg2, arg3, arg4, ...
                arg5, arg6) ;
        case 7
            [C_opaque, kind] = gbmex_extract (1, arg1, arg2, arg3, arg4, ...
                arg5, arg6, arg7) ;
    end
    C = gb_mexfunction_result (1, C_opaque, kind) ;
end

