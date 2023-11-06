function finalize
%GRB.FINALIZE finalize SuiteSparse:GraphBLAS.
%
%   GrB.finalize
%
% GrB.finalize clears the global settings of GraphBLAS.
% Its use is optional in this version of SuiteSparse:GraphBLAS.
%
% See also GrB.clear, GrB.init.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% does not call GrB_finalize
gbclear ;

