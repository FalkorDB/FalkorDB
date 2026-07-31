function C = assign (arg1, arg2, arg3, arg4, arg5, arg6, arg7)
%GHB.ASSIGN assign a submatrix into a matrix.
%
% syntax for a new matrix C:                        computation:
% C = GhB.assign (Cin, A, I, J, desc)               % C = Cin ; C(I,J) = A
% C = GhB.assign (Cin, accum, A, I, J, desc)        % C = Cin ; C(I,J) += A
% C = GhB.assign (Cin, M, A, I, J, desc)            % C = Cin ; C<M>(I,J) = A
% C = GhB.assign (Cin, M, accum, A, I, J, desc)     % C = Cin ; C<M>(I,J) += A
%
% in-place syntax:
% GhB.assign (C, A, I, J, desc)                     % C(I,J) = A
% GhB.assign (C, accum, A, I, J, desc)              % C(I,J) += A
% GhB.assign (C, M, A, I, J, desc)                  % C<M>(I,J) = A
% GhB.assign (C, M, accum, A, I, J, desc)           % C<M>(I,J) += A
%
% Cin (or C) and A are required parameters.  All others are optional.  The
% arguments are parsed according to their type.  Arguments with different types
% can appear in any order:
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
%   0:  { }     This is ':', like C(:,J), refering to all m
%               rows, if C is m-by-n.
%
%   1:  { I }   1D list of row indices, like C(I,J).
%
%   2:  { start,fini }  start and fini are scalars, defining I = start:fini.
%
%   3:  { start,inc,fini } start, inc, and fini are scalars,
%               defining I = start:inc:fini.
%
% The J argument is identical, except that it is a list of column indices of C.
% If only one cell array is provided, J = {  } is implied, refering to all n
% columns of C, like C(I,:).  GhB.assign does not support linear indexing of a
% 2D matrix, as in C(I)=A when C is a 2D matrix.
%
% If neither I nor J are provided on input, then this implies both I = { } and
% J = { }, or C(:,:), refering to all rows and columns of C.
%
% desc.base modifies how I, start, and fini are interpretted.  If desc.base is
% 'zero-based' then they are interpretted as zero-based indices, where 0 is the
% first row or column.  If desc.base is 'one-based' (which is the default),
% then indices are intrepetted as 1-based.
%
% A: this argument either has size length(I)-by-length(J) (or A' if d.in0
%   is 'transpose'), or it is 1-by-1 for scalar assignment (like C(1:2,1:2)=pi,
%   which assigns the scalar pi to the leading 2-by-2 submatrix of C).  For
%   scalar assignment, A must contain an entry; it cannot be empty (for
%   example, A = sparse (0)).
%
% accum: an optional binary operator, defined by a string ('+.double') for
%   example.  This allows for C(I,J) = C(I,J) + A to be computed.  If not
%   present, no accumulator is used and C(I,J)=A is computed.  In the
%   computations listed above it is shown as "+=" but any binary operator may
%   be used.  See 'help GrB.binopinfo' for available binary operators.
%
% M: an optional mask matrix, the same size as C.
%
% C or Cin: a required input matrix, containing the initial content of the
% matrix C.  For the in-place syntax, the GhB matrix C is modified in-place.
%
% Except for C for the inplace syntax, all input matrices may be either
% GraphBLAS/built-in matrices, in any combination.  C is returned as a
% GraphBLAS GhB matrix.
%
% Example:
%
%   A = sprand (5, 4, 0.5)
%   AT = A'
%   M = sparse (rand (4, 5)) > 0.5
%   Cin = sprand (4, 5, 0.5)
%
%   d.in0 = 'transpose'
%   d.mask = 'complement'
%   C = GhB (Cin) ;
%   GhB.assign (C, M, A, d)
%   C2 = Cin ;
%   C2 (~M) = AT (~M) ;
%   C2 - sparse (C)
%
%   I = [2 1 5]
%   J = [3 3 1 2]
%   B = sprandn (length (I), length (J), 0.5)
%   Cin = sprand (6, 3, 0.5)
%   C = GhB (Cin) ;
%   GhB.assign (C, B, {I}, {J}) ;
%   C2 = Cin ;
%   C2 (I,J) = B ;
%   C2 - sparse (C)
%
% See also GrB.subassign, GrB/subsasgn, GrB.binopinfo.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

narginchk (2, 7) ;

if (gb_is_grb (arg1))
    arg1 = struct (arg1) ;
end

if (gb_is_grb (arg2))
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
            gbmex_assign (1, arg1, arg2) ;
        case 3
            gbmex_assign (1, arg1, arg2, arg3) ;
        case 4
            gbmex_assign (1, arg1, arg2, arg3, arg4) ;
        case 5
            gbmex_assign (1, arg1, arg2, arg3, arg4, arg5) ;
        case 6
            gbmex_assign (1, arg1, arg2, arg3, arg4, arg5, arg6) ;
        case 7
            gbmex_assign (1, arg1, arg2, arg3, arg4, arg5, arg6, arg7) ;
    end
else
    switch (nargin)
        case 2
            [C_opaque, kind] = gbmex_assign (1, arg1, arg2) ;
        case 3
            [C_opaque, kind] = gbmex_assign (1, arg1, arg2, arg3) ;
        case 4
            [C_opaque, kind] = gbmex_assign (1, arg1, arg2, arg3, arg4) ;
        case 5
            [C_opaque, kind] = gbmex_assign (1, arg1, arg2, arg3, arg4, arg5) ;
        case 6
            [C_opaque, kind] = gbmex_assign (1, arg1, arg2, arg3, arg4, ...
                arg5, arg6) ;
        case 7
            [C_opaque, kind] = gbmex_assign (1, arg1, arg2, arg3, arg4, ...
                arg5, arg6, arg7) ;
    end
    C = gb_mexfunction_result (1, C_opaque, kind) ;
end

