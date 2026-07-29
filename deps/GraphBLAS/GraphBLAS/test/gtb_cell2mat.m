function [varargout] = gtb_cell2mat (ghb, varargin)
%GTB_CELL2MAT wrapper for GrB.cell2mat and GhB.cell2mat

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (~(ghb == 0 || ghb == 1))
    ghb = rand (1) > 0.5 ; % choose ghb at random
end
if (ghb)
    [varargout{1:nargout}] = GhB.cell2mat (varargin {:}) ;
else
    [varargout{1:nargout}] = GrB.cell2mat (varargin {:}) ;
end

