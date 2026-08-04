function C = load (filename)
%GHB.LOAD Load a single GraphBLAS matrix from a file.
% C = GhB.load (filename) loads a single GhB or GhB matrix from a file.  If
% the filename is not present, it defaults to 'GrB_Matrix.mat'.
%
% GhB.load can load in *.mat files created by GhB.save or GrB.save from this or
% earlier versions of GraphBLAS.
%
% NOTE: As of GraphBLAS v10.4.0, this method is no longer needed in MATLAB;
% just MATLAB load/save methods instead.  Octave cannot load/save the GhB and
% GhB objects, so this method is useful for Octave.
%
% Examples:
%
%   A = GhB.random (4, 4, 0.5)
%   GhB.save (A) ;              % A can be a GhB or built-in matrix
%   clear all
%   A = GhB.load ('A.mat') ;    % A is now a GhB matrix
%
%   % saving a matrix expression
%   GhB.save (2*A-1)            % save a matrix computation to GhB_Matrix.mat
%   GhB.load                    % load it back in
%
% See also load, save, GhB.save, GhB.serialize, GhB.deserialize.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin < 1)
    filename = 'GrB_Matrix.mat' ;
end

C = gb_load (1, filename) ;

