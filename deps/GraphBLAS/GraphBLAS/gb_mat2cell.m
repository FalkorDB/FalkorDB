function C = gb_mat2cell (ghb, A, varargin)
%GB_MAT2CELL implements GrB/mat2cell and GhB/mat2cell.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

m = int64 (varargin {1}) ;
if (nargin < 4)
    n = size (A, 2) ;
else
    n = int64 (varargin {2}) ;
end

% C is returned as a cell array of GrB or GhB objects
C = gzb_split (ghb, A, m, n) ;

