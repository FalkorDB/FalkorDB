function display (G) %#ok<DISPLAY>
%DISPLAY display the contents of a GraphBLAS matrix.
% display (G) displays the attributes and first few entries of a
% GraphBLAS sparse matrix object.  Use disp(G,3) to display all of the
% content of G.
%
% See also GrB/disp.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

name = inputname (1) ;
if (~isempty (name))
    fprintf ('\n%s =\n', name) ;
end
G = G.opaque ;
gbdisp (G, gb_nnz (G), 2) ;
fprintf ('\n') ;

