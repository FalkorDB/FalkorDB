function print (A, level)
%PRINT display the contents of a matrix.
% GrB.print (A, level) displays the matrix A.  The 2nd argument controls how
% much is printed; 0: none, 1: terse, 2: a few entries, 3: all, 4: a few
% entries with high precision, 5: all with high precision.  The default is 2 if
% level is not present.
%
% This method is identical to the overloaded disp method, but appears as a
% static method (GrB.print or GhB.print) which allows it to be used on built-in
% matrices.
%
% Example:
%
%   A = sprand (50, 50, 0.1) ;
%   % just print a few entries
%   GrB.print (A, 2)
%   G = GrB (A)
%   % print all entries
%   A
%   GrB.print (G, 3)
%   % print all entries in full precision
%   format long
%   A
%   GrB.print (G, 5)
%
% See also GrB/disp, GrB/display.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin < 2)
    gb_display ('', A) ;
else
    gb_display ('', A, level) ;
end

