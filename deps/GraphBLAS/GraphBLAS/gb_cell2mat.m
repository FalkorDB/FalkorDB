function C = gb_cell2mat (ghb, A)
%GB_CELL2MAT implements GrB/cell2mat and GhB/cell2mat.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (~iscell (A))
    error ('GrB:error', 'input must be a cell array') ;
end
if (ndims (A) > 2) %#ok<ISMAT>
    error ('GrB:error', 'only 2D cell arrays are supported') ;
end

C = gzb_cat (ghb, A) ;

