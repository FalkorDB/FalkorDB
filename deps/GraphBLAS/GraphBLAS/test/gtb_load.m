function [varargout] = gtb_load (ghb, varargin)
%GTB_LOAD wrapper for GrB.load and GhB.load

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (~(ghb == 0 || ghb == 1))
    ghb = rand (1) > 0.5 ; % choose ghb at random
end
if (ghb)
    [varargout{1:nargout}] = GhB.load (varargin {:}) ;
else
    [varargout{1:nargout}] = GrB.load (varargin {:}) ;
end

