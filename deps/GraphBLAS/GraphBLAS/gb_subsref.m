function C = gb_subsref (ghb, A, S)
%GB_SUBREF implements C=A(I,J) or C=A(I) for GrB and GhB.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% FUTURE: add all forms of linear indexing.

if (gb_is_grb (A))
    A = struct (A) ;
end

[m, n] = gbmex_size (A) ;

if (length (S) > 1)
    error ('GrB:error', 'nested indexing not supported') ;
end

if (~isequal (S.type, '()'))
    error ('GrB:error', 'index type %s not supported', S.type) ;
end

ndims = length (S.subs) ;

if (ndims == 1)

    % C = A(M) if M is logical, or C=A(I) otherwise
    S1 = S.subs {1} ;
    if (gb_is_grb (S1))
        S1 = struct (S1) ;
    end
    if (isequal (gbmex_type (S1), 'logical'))
        % C = A (M) for logical indexing
        C = gzb_logextract (ghb, A, S1) ;
    else
        % C = A (I)
        [I, whole] = gb_index (S1) ;
        if (m == 1 || n == 1)
            % C = A (I) for a vector A
            if (m > 1)
                C = gzb_extract (ghb, A, I, { }) ;
            else
                C = gzb_extract (ghb, A, { }, I) ;
            end
            [cm, ~] = gb_size (C) ;
            if (whole && cm == 1)
                C = gzb_trans (ghb, C) ;
            end
        else
            % C = A (I) for a matrix A
            if (whole)
                % C = A (:), whole matrix case
                [~, mn] = gb_2d_to_1d (0, 0, m, n) ;
                C = gzb_reshape (ghb, A, mn, 1, true) ;
            else
                % C = A (I), general case not yet supported
                error ('GrB:error', ...
                    'Except for C=A(:), linear indexing not yet supported') ;
            end
        end
    end

elseif (ndims == 2)

    % C = A (I,J)
    I = gb_index (S.subs {1}) ;
    J = gb_index (S.subs {2}) ;
    C = gzb_extract (ghb, A, I, J) ;

else

    % sparse N-dimensional arrays for N > 2 will not be supported
    error ('GrB:error', '%dD indexing not supported', ndims) ;

end

