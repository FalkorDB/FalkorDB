function t = timing (c)
%GRB.TIMING get/set the GraphBLAS diagnostic timings
%
%   t = GrB.timing ;      % get the current timings, do not clear them.
%   t = GrB.timing (0) ;  % get the current timings, and clear them.
%
% This method is for development only, and returns t as all zero when
% GraphBLAS is compiled for production use.  GraphBLAS must be compiled
% with -DGB_TIMING for timers to be enabled.
%
% See also spparms.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    t = gbtiming ;
else
    t = gbtiming (c) ;
end

