function [p, varargout] = dmperm (G)
%DMPERM Dulmage-Mendelsohn permutation.
% See 'help dmperm' for details.
%
% See also GrB/amd, GrB/colamd.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[p, varargout{1:nargout-1}] = dmperm (logical (G)) ;

