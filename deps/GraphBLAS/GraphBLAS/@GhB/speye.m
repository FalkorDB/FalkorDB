function C = speye (varargin)
%GHB.SPEYE sparse identity matrix.
% C = GhB.speye (...) is identical to GhB.eye; see 'help GhB.eye' for details.
%
% See also GhB.eye.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_speye (1, 'speye', varargin {:}) ;

