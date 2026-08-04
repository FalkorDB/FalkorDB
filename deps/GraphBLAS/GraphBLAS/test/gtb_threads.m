function [varargout] = gtb_threads (ghb, varargin)
%GTB_THREADS wrapper for GrB.threads and GhB.threads

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (~(ghb == 0 || ghb == 1))
    ghb = rand (1) > 0.5 ; % choose ghb at random
end
if (ghb)
    [varargout{1:nargout}] = GhB.threads (varargin {:}) ;
else
    [varargout{1:nargout}] = GrB.threads (varargin {:}) ;
end

