function C = speye (varargin)
%GRB.SPEYE sparse identity matrix.
% C = GrB.speye (...) is identical to GrB.eye; see 'help GrB.eye' for
% details.
%
% See also GrB.eye.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = GrB (gb_speye ('speye', varargin {:})) ;

