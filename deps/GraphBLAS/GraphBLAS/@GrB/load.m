function C = load (filename)
%GRB.LOAD Load a single GraphBLAS matrix from a file.
% C = GrB.load (filename) loads a single GrB or GhB matrix from a file.  If
% the filename is not present, it defaults to 'GrB_Matrix.mat'.
%
% GrB.load can load in *.mat files created by GhB.save or GrB.save from this or
% earlier versions of GraphBLAS.
%
% NOTE: As of GraphBLAS v10.4.0, this method is no longer needed in MATLAB;
% just MATLAB load/save methods instead.  Octave cannot load/save the GrB and
% GhB objects, so this method is useful for Octave.
%
% Examples:

%
%   A = GrB.random (4, 4, 0.5)
%   GrB.save (A) ;              % A can be a GrB or built-in matrix
%   clear all
%   A = GrB.load ('A.mat') ;    % A is now a GrB matrix
%
%   % saving a matrix expression
%   GrB.save (2*A-1)            % save a matrix computation to GrB_Matrix.mat
%   GrB.load                    % load it back in
%
% See also load, save, GrB.save, GrB.serialize, GrB.deserialize.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin < 1)
    filename = 'GrB_Matrix.mat' ;
end

C = gb_load (0, filename) ;

