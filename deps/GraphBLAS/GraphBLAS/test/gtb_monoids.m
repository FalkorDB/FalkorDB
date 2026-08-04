function [varargout] = gtb_monoids (ghb, varargin)
%GTB_MONOIDS wrapper for GrB.monoids and GhB.monoids

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (~(ghb == 0 || ghb == 1))
    ghb = rand (1) > 0.5 ; % choose ghb at random
end
if (ghb)
    [varargout{1:nargout}] = GhB.monoids (varargin {:}) ;
else
    [varargout{1:nargout}] = GrB.monoids (varargin {:}) ;
end

