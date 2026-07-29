function [varargout] = gtb_nonz (ghb, varargin)
%GTB_NONZ wrapper for GrB.nonz and GhB.nonz

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (~(ghb == 0 || ghb == 1))
    ghb = rand (1) > 0.5 ; % choose ghb at random
end
if (ghb)
    [varargout{1:nargout}] = GhB.nonz (varargin {:}) ;
else
    [varargout{1:nargout}] = GrB.nonz (varargin {:}) ;
end

