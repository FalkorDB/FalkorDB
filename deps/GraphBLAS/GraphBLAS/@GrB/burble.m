function b = burble (b_arg)
%GRB.BURBLE get/set the GraphBLAS burble option.
%
%   b = GrB.burble ;      % get the current burble
%   GrB.burble (b) ;      % set the burble
%
% GrB.burble gets and/or sets the burble setting, which controls diagnostic
% output in GraphBLAS.
%
% See also spparms.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    b = gbmex_burble ;
else
    b = gbmex_burble (b_arg) ;
end

