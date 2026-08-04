function S = struct (G)
%STRUCT return the opaque contents of a GrB or GhB object.
% This method is meant for testing, development, and internal use only.
% It is not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

S = G.opaque ;
