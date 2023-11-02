function disp (A, level)
%DISP display the contents of a matrix.
% disp (A, level) displays the matrix A.  The 2nd argument controls how
% much is printed; 0: none, 1: terse, 2: a few entries, 3: all, 4: a few
% entries with high precision, 5: all with high precision.  The default is
% 2 if level is not present.  To use this function on a built-in sparse
% matrix, use disp (A, GrB (level)).  This is useful since disp(A) will
% always display all entries of A, which can be too verbose if nnz (A)
% is huge.
%
% Example:
%
%   A = sprand (50, 50, 0.1) ;
%   % just print a few entries
%   disp (A, GrB (2))
%   G = GrB (A)
%   % print all entries
%   A
%   disp (G, 3)
%   % print all entries in full precision
%   format long
%   A
%   disp (G, 5)
%
% See also GrB/display.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin < 2)
    level = 2 ;
else
    level = gb_get_scalar (level) ;
end

if (level > 0)
    name = inputname (1) ;
    if (~isempty (name))
        fprintf ('\n%s =\n', name) ;
    end
end

if (isobject (A))
    A = A.opaque ;
    gbdisp (A, gb_nnz (A), level) ;
else
    gbdisp (A, nnz (A), level) ;
end

if (level > 0)
    fprintf ('\n') ;
end

