function [varargout] = gtb_semirings (ghb, varargin)
%GTB_SEMIRINGS wrapper for GrB.semirings and GhB.semirings

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (~(ghb == 0 || ghb == 1))
    ghb = rand (1) > 0.5 ; % choose ghb at random
end
if (ghb)
    [varargout{1:nargout}] = GhB.semirings (varargin {:}) ;
else
    [varargout{1:nargout}] = GrB.semirings (varargin {:}) ;
end

