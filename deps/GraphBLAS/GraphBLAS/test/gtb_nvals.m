function [varargout] = gtb_nvals (ghb, varargin)
%GTB_NVALS wrapper for GrB.nvals and GhB.nvals

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (~(ghb == 0 || ghb == 1))
    ghb = rand (1) > 0.5 ; % choose ghb at random
end
if (ghb)
    [varargout{1:nargout}] = GhB.nvals (varargin {:}) ;
else
    [varargout{1:nargout}] = GrB.nvals (varargin {:}) ;
end

