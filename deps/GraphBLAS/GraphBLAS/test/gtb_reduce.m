function [varargout] = gtb_reduce (ghb, varargin)
%GTB_REDUCE wrapper for GrB.reduce and GhB.reduce

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (~(ghb == 0 || ghb == 1))
    ghb = rand (1) > 0.5 ; % choose ghb at random
end
if (ghb)
    [varargout{1:nargout}] = GhB.reduce (varargin {:}) ;
else
    [varargout{1:nargout}] = GrB.reduce (varargin {:}) ;
end

