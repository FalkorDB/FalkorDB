function [varargout] = gtb_apply2 (ghb, varargin)
%GTB_APPLY2 wrapper for GrB.apply2 and GhB.apply2

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (~(ghb == 0 || ghb == 1))
    ghb = rand (1) > 0.5 ; % choose ghb at random
end
if (ghb)
    [varargout{1:nargout}] = GhB.apply2 (varargin {:}) ;
else
    [varargout{1:nargout}] = GrB.apply2 (varargin {:}) ;
end

