function C = subsasgn (C, S, A)
%SUBSASGN C(I,J) = A or C(I) = A; assign submatrix.
% C(I,J) = A assigns A into the C(I,J) submatrix of the GraphBLAS matrix C.  A
% must be either a matrix of size length(I)-by-length(J), or a scalar.  Note
% that C(I,J) = 0 differs from C(I,J) = sparse (0).  The former places an
% explicit entry with value zero in all positions of C(I,J).  The latter
% deletes all entries in C(I,J).  With a built-in sparse matrix C, both
% statements delete all entries in C(I,J) since Built-in sparse matrices never
% include explicit zeros.
%
% Linear indexing is not fully yet supported.  With a single index, C(I) = A,
% both C and A must be vectors, except for C(:) = A where A is a matrix and C
% is a vector.
%
% If M is a logical matrix, C (M) = x is an assignment via logical indexing,
% where C and M have the same size, and x(:) is either a vector of length
% nnz (M), or a scalar.
%
% C (M) = A (M), where the logical matrix M is used on both the sides of the
% assignment, is the same as C = GhB.subassign (C, M, A).  If C and A (or M)
% are GraphBLAS matrices, C (M) = A (M) uses GraphBLAS via operator
% overloading.  The statement C (M) = A (M) takes about twice the time as C =
% GhB.subassign (C, M, A), so the latter is preferred for best performance.
% However, both methods in GraphBLAS are many thousands of times faster than
% C (M) = A (M) using purely built-in sparse matrices C, M, and A, when the
% matrices are large.
%
% If I or J are very large colon notation expressions, then C(I,J) = A is not
% possible, because I and J are created as explicit lists first, before passing
% them to GraphBLAS.  See GhB.subassign instead.  See also the example with
% 'help GhB.extract'.
%
% Just as the built-in C(I,J) = A, the GraphBLAS assignment can change the size
% of C if the indices I and J extend past the current dimesions of C.
%
% See also GhB/subsref, GhB/subsindex, GhB.assign, GhB.subassign.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% FUTURE: add all forms of linear indexing.

if (gb_is_grb (A))
    A = struct (A) ;
end

if (~isequal (S.type, '()'))
    error ('GrB:error', 'index type %s not supported', S.type) ;
end

ndims = length (S.subs) ;

if (ndims == 1)

    % C (M) = A if M is logical, or C (I) = A otherwise
    S1 = S.subs {1} ;
    if (gb_is_grb (S1))
        S1 = struct (S1) ;
    end
    if (isequal (gbmex_type (S1), 'logical'))
        % C (M) = A for logical assignment (where M is S1)
        [am, an] = gbmex_size (A) ;
        if (am == 1 && an == 1)
            % C (M) = scalar
            gbmex_subassign (1, C, S1, A) ;
        else
            % C (M) = A where A is a vector
            gbmex_logassign (1, C, S1, A) ;
        end
    else
        % C (I) = A
        [cm, cn] = gbmex_size (C) ;
        [I, whole] = gb_index (S1) ;
        if (cm == 1 || cn == 1)
            % C (I) = A for a vector or scalar C
            gbmex_subassign (1, C, I, A) ;
        else
            if (whole)
                [am, an] = gbmex_size (A) ;
                if (am == 1 && an == 1)
                    % C (:) = scalar, the same as C (:,:) = scalar.
                    % C becomes an iso full matrix
                    C = GhB (cm, cn, gbmex_type (C), gbmex_format (C)) ;
                    gbmex_subassign (1, C, { }, { }, A) ;
                else
                    % C (:) = A for a matrix C and vector A
                    C = GhB (gbmex_reshape (1, A, cm, cn, true)) ;
                end
            else
                % C (I) = A, general case not yet supported
                error ('GrB:error', ...
                    'Except for C(:)=A, linear indexing not yet supported') ;
            end
        end
    end

elseif (ndims == 2)

    % C (I,J) = A where A is length(I)-by-length(J), or a scalar
    I = gb_index (S.subs {1}) ;
    J = gb_index (S.subs {2}) ;
    gbmex_subassign (1, C, I, J, A) ;

else

    % sparse N-dimensional arrays for N > 2 will not be supported
    error ('GrB:error', '%dD indexing not supported', ndims) ;

end

