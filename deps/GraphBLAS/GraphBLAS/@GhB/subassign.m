function C = subassign (arg1, arg2, arg3, arg4, arg5, arg6, arg7)
%GHB.SUBASSIGN assign a submatrix into a matrix.
%
% syntax for a new matrix C:                        computation:
% C = GhB.subassign (Cin, A, I, J, desc)            % C = Cin ; C(I,J) = A
% C = GhB.subassign (Cin, accum, A, I, J, desc)     % C = Cin ; C(I,J) += A
% C = GhB.subassign (Cin, M, A, I, J, desc)         % C = Cin ; C(I,J)<M> = A
% C = GhB.subassign (Cin, M, accum, A, I, J, desc)  % C = Cin ; C(I,J)<M> += A
%
% in-place syntax:
% GhB.subassign (C, A, I, J, desc)                  % C(I,J) = A
% GhB.subassign (C, accum, A, I, J, desc)           % C(I,J) += A
% GhB.subassign (C, M, A, I, J, desc)               % C(I,J)<M> = A
% GhB.subassign (C, M, accum, A, I, J, desc)        % C(I,J)<M> += A
%
% GhB.subassign is identical to GhB.assign, with two key differences:
%
% (1) The mask is different.  With GhB.subassign, the mask M is
%       length(I)-by-length(J), and M(i,j) controls how A(i,j) is assigned into
%       C(I(i),J(j)).  With GhB.assign, the mask M has the same size as C, and
%       M(i,j) controls how C(i,j) is assigned.
% (2) The d.out = 'replace' option differs.  GhB.assign can clear
%       entries outside the C(I,J) submatrix; GhB.subassign cannot.
%
% If there is no mask, or if I and J are ':', then the two methods are
% identical.  The examples shown in 'help GhB.assign' also work with
% GhB.subassign.  Otherwise, GhB.subassign is faster.  The two methods are
% described below, where '+' is the optional accum operator.
%
%   step  | GhB.assign      GhB.subassign
%   ----  | ----------      -------------
%   1     | S = C(I,J)      S = C(I,J)
%   2     | S = S + A       S<M> = S + A
%   3     | Z = C           C(I,J) = S
%   4     | Z(I,J) = S
%   5     | C<M> = Z
%
% Refer to GhB.assign for more details.
%
% See also GhB.assign, GhB/subsasgn, GrB.binopinfo.

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
            gbmex_subassign (1, arg1, arg2) ;
        case 3
            gbmex_subassign (1, arg1, arg2, arg3) ;
        case 4
            gbmex_subassign (1, arg1, arg2, arg3, arg4) ;
        case 5
            gbmex_subassign (1, arg1, arg2, arg3, arg4, arg5) ;
        case 6
            gbmex_subassign (1, arg1, arg2, arg3, arg4, arg5, arg6) ;
        case 7
            gbmex_subassign (1, arg1, arg2, arg3, arg4, arg5, arg6, arg7) ;
    end
else
    switch (nargin)
        case 2
            [C_opaque, kind] = gbmex_subassign (1, arg1, arg2) ;
        case 3
            [C_opaque, kind] = gbmex_subassign (1, arg1, arg2, arg3) ;
        case 4
            [C_opaque, kind] = gbmex_subassign (1, arg1, arg2, arg3, arg4) ;
        case 5
            [C_opaque, kind] = gbmex_subassign (1, arg1, arg2, arg3, arg4, ...
                arg5) ;
        case 6
            [C_opaque, kind] = gbmex_subassign (1, arg1, arg2, arg3, arg4, ...
                arg5, arg6) ;
        case 7
            [C_opaque, kind] = gbmex_subassign (1, arg1, arg2, arg3, arg4, ...
                arg5, arg6, arg7) ;
    end
    C = gb_mexfunction_result (1, C_opaque, kind) ;
end

