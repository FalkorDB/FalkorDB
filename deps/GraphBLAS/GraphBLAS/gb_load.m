function C = gb_load (ghb, filename)
%GB_LOAD implements GrB.load and GhB.load.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

S = load (filename) ;

if (isfield (S, 'GraphBLAS_struct_from_GrB_save'))
    % S was created by GrB.save from GraphBLAS v10.3.1 or earlier
    C = gzb_loadhistorical (ghb, S.GraphBLAS_struct_from_GrB_save) ;
elseif (isfield (S, 'GrB_Matrix_from_GrB_save'))
    % S was created by GrB.save from GraphBLAS v10.4.0 or later,
    % and it already contains a properly loaded GrB matrix.
    C = S.GrB_Matrix_from_GrB_save ;
else
    % S has already been properly loaded by GrB/loadobj or GhB/loadobj
    C = S ;
end

