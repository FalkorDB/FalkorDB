function v = ver
%GRB.VER Version information for GraphBLAS
% v = GrB.ver returns a struct with the SuiteSparse:GraphBLAS version.
% With no outputs, the version information is displayed.
%
% See also ver, version, GrB.version.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargout == 0)
    gbver ;
else
    v = gbver ;
end

