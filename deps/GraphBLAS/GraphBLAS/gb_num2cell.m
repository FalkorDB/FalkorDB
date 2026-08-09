function C = gb_num2cell (ghb, A, dim)
%GB_NUM2CELL implements num2cell for GrB and GhB.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 3 && isequal (dim, [1 2]))

    % whole matrix, not transposed
    C = { A } ;

elseif (nargin == 3 && isequal (dim, [2 1]))

    % whole matrix, transposed
    C = { A.' } ;

else

    % split into scalars, rows, or columns
    if (isobject (A))
        if (gb_is_grb (A))
            A = struct (A) ;
        end
        [m, n] = gbmex_size (A) ;
    else
        [m, n] = size (A) ;
    end

    if (nargin == 2)
        % split A into scalars
        C = gzb_split (ghb, A, ones (m, 1), ones (n, 1)) ;
    elseif (isequal (dim, 1))
        % split A into columns
        C = gzb_split (ghb, A, m, ones (n, 1)) ;
    elseif (isequal (dim, 2))
        % split A into rows
        C = gzb_split (ghb, A, ones (m, 1), n) ;
    else
        error ('GrB:error', 'unknown option') ;
    end

end

