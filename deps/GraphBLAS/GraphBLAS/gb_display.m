function gb_display (name, A, level)
%GB_DISPLAY display the contents of a matrix.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (A))
    A = struct (A) ;
end

if (nargin < 3)
    k = 2 ;
else
    k = gb_get_scalar (level) ;
end

if (~isempty (name))
    % the builtin disp method does not print the name of the matrix, while
    % the builtin display method does.
    fprintf ('\n%s =\n', name) ;
end

gbmex_disp (A, k) ;

if (k > 1)
    fprintf ('\n') ;
end

