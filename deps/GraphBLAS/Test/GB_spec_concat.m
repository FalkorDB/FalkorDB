function C = GB_spec_concat (Tiles, ctype)
%GB_SPEC_CONCAT a mimic of GxB_Matrix_concat
%
% Usage:
% C = GB_spec_concat (Types, ctype)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

%-------------------------------------------------------------------------------
% get inputs
%-------------------------------------------------------------------------------

if (nargout > 1 || nargin ~= 2)
    error ('usage: C = GB_spec_concat (Tiles, ctype)') ;
end

% get the tiles and typecast them to the type of C
if (~iscell (Tiles))
    error ('Tiles must be a cell array') ;
end
[m, n] = size (Tiles) ;
Tiles_matrix  = cell (m,n) ;
Tiles_pattern = cell (m,n) ;
for i = 1:m
    for j = 1:n
        A = GB_spec_matrix (Tiles {i,j}) ;
        A_matrix = A.matrix ;
        if (~isequal (A.class, ctype))
            % typecast the tile into ctype
            A_matrix = GB_mex_cast (A_matrix, ctype) ;
        end
        Tiles_matrix  {i,j} = A_matrix ;
        Tiles_pattern {i,j} = A.pattern ;
    end
end

%-------------------------------------------------------------------------------
% C = concat (Tiles)
%-------------------------------------------------------------------------------

C.matrix  = cell2mat (Tiles_matrix) ;
C.pattern = cell2mat (Tiles_pattern) ;
C.class = ctype ;

