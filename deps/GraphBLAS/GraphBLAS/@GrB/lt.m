function C = lt (A, B)
%A < B less than.
% C = (A < B) compares A and B element-by-element.  One or
% both may be scalars.  Otherwise, A and B must have the same size.
%
% See also GrB/le, GrB/gt, GrB/ge, GrB/ne, GrB/eq.

% The pattern of C depends on the type of inputs:
% A scalar, B scalar:  C is scalar.
% A scalar, B matrix:  C is full if A<0, otherwise C is a subset of B.
% B scalar, A matrix:  C is full if B>0, otherwise C is a subset of A.
% A matrix, B matrix:  C has the pattern of the set union, A+B.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (isobject (A))
    A = A.opaque ;
end

if (isobject (B))
    B = B.opaque ;
end

[am, an, atype] = gbsize (A) ;
[bm, bn, btype] = gbsize (B) ;
a_is_scalar = (am == 1) && (an == 1) ;
b_is_scalar = (bm == 1) && (bn == 1) ;
ctype = gboptype (atype, btype) ;

if (a_is_scalar)
    if (b_is_scalar)
        % both A and B are scalars
        C = GrB (gbeunion (A, 0, '<', B, 0)) ;
    else
        % A is a scalar, B is a matrix
        if (gb_scalar (A) < 0)
            if (~gb_issigned (btype))
                % a < 0, and B has an unsigned type.  C is all true.
                C = GrB (gb_scalar_to_full (bm, bn, 'logical', ...
                    gb_fmt (B), true)) ;
            else
                % since a < 0, entries not present in B result in a true
                % value, so the result is full.  Expand A to full.
                A = gb_scalar_to_full (bm, bn, ctype, gb_fmt (B), A) ;
                C = GrB (gbemult (A, '<', gbfull (B, ctype))) ;
            end
        else
            % since a >= 0, entries not present in B result in a false
            % value, so the result is a sparse subset of B.  select all
            % entries in B > a, then convert to true.
            C = GrB (gbapply ('1.logical', gbselect (B, '>', A))) ;
        end
    end
else
    if (b_is_scalar)
        % A is a matrix, B is a scalar
        b = gb_scalar (B) ;
        if (b < 0 && ~gb_issigned (atype))
            % b is negative, and A has an unsigned type.  C is all false.
            C = GrB (gbnew (am, an, 'logical')) ;
        elseif (b > 0)
            % since b > 0, entries not present in A result in a true
            % value, so the result is full.  Expand B to a full matrix.
            B = gb_scalar_to_full (am, an, ctype, gb_fmt (A), B) ;
            C = GrB (gbemult (gbfull (A, ctype), '<', B)) ;
        else
            % since b <= 0, entries not present in A result in a false
            % value, so the result is a sparse subset of A.  Select all
            % entries in A < b, then convert to true.
            C = GrB (gbapply ('1.logical', gbselect (A, '<', B))) ;
        end
    else
        % both A and B are matrices.  C is the set union of A and B.
        C = GrB (gbeunion (A, 0, '<', B, 0)) ;
    end
end

