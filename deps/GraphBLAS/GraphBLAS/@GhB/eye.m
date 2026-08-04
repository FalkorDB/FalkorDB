function C = eye (varargin)
%GHB.EYE sparse identity matrix.
% C = GhB.eye (n) creates a sparse n-by-n identity matrix of type 'double'.
% C = GhB.eye (m,n) or GhB.eye ([m n]) is an m-by-n identity matrix.
%
% C = GhB.eye (m,n,type) or GhB.eye ([m n],type) creates a sparse m-by-n
% identity matrix C of the given GraphBLAS type, either 'double', 'single',
% 'logical', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32',
% 'uint64', 'single complex', or 'double complex'.
%
% See also GhB/spones, spdiags, GhB.speye, GhB.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_speye (1, 'eye', varargin {:}) ;

