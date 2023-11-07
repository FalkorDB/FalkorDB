function C = eye (varargin)
%GRB.EYE sparse identity matrix.
% C = GrB.eye (n) creates a sparse n-by-n identity matrix of type 'double'.
% C = GrB.eye (m,n) or GrB.eye ([m n]) is an m-by-n identity matrix.
%
% C = GrB.eye (m,n,type) or GrB.eye ([m n],type) creates a sparse m-by-n
% identity matrix C of the given GraphBLAS type, either 'double', 'single',
% 'logical', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16',
% 'uint32', 'uint64', 'single complex', or 'double complex'.
%
% See also GrB/spones, spdiags, GrB.speye, GrB.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = GrB (gb_speye ('eye', varargin {:})) ;

