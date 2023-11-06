function C = load (filename)
%GRB.LOAD Load a single GraphBLAS matrix from a file.
% C = GrB.load (filename) loads a single @GrB matrix from a file.  The file
% must have been previously created by GrB.save.  If the filename is not
% present, it defaults to 'GrB_Matrix.mat'.
%
% Examples:
%
%   A = GrB.random (4, 4, 0.5)
%   GrB.save (A) ;              % A can be a @GrB or built-in matrix
%   clear all
%   A = GrB.load ('A.mat') ;    % A is now a @GrB matrix
%
%   % saving a matrix expression
%   GrB.save (2*A-1)            % save a matrix computation to GrB_Matrix.mat
%   GrB.load                    % load it back in
%
% See also GrB.save, GrB/struct, GrB.serialize, GrB.deserialize.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin < 1)
    filename = 'GrB_Matrix.mat' ;
end

% load in the opaque struct from the file
S = load (filename, 'GraphBLAS_struct_from_GrB_save') ;

% convert it to a @GrB object
C = GrB (S.GraphBLAS_struct_from_GrB_save) ;

