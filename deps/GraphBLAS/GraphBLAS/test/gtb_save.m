function [varargout] = gtb_save (ghb, varargin)
%GTB_SAVE wrapper for GrB.save and GhB.save

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (~(ghb == 0 || ghb == 1))
    ghb = rand (1) > 0.5 ; % choose ghb at random
end
if (ghb)
    [varargout{1:nargout}] = GhB.save (varargin {:}) ;
else
    [varargout{1:nargout}] = GrB.save (varargin {:}) ;
end

