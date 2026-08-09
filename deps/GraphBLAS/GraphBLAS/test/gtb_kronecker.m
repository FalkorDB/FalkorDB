function [varargout] = gtb_kronecker (ghb, varargin)
%GTB_KRONECKER wrapper for GrB.kronecker and GhB.kronecker

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (~(ghb == 0 || ghb == 1))
    ghb = rand (1) > 0.5 ; % choose ghb at random
end
if (ghb)
    [varargout{1:nargout}] = GhB.kronecker (varargin {:}) ;
else
    [varargout{1:nargout}] = GrB.kronecker (varargin {:}) ;
end

