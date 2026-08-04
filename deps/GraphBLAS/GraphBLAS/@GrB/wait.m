function wait (A)
%WAIT finish work on a GrB or GhB matrix
% GrB.wait or GhB.wait is only needed for GhB matrix inputs.  Either example
% below finishes a GhB matrix A, and does not modify A if it is GrB matrix.
%
% Example:
%
%   GrB.wait (A) ;
%   GhB.wait (A) ;
%
% See also GrB.clear, GrB.finalize.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

gbmex_wait (A) ;

