function s = gtb_tricount (ghb, varargin)
%GTB_TRICOUNT wrapper for GrB.tricount and GhB.tricount

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (~(ghb == 0 || ghb == 1))
    ghb = rand (1) > 0.5 ; % choose ghb at random
end
if (ghb)
    s = GhB.tricount (varargin {:}) ;
else
    s = GrB.tricount (varargin {:}) ;
end

