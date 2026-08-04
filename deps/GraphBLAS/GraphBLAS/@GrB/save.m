function filename_used = save (GrB_Matrix_from_GrB_save, filename)
%GRB.SAVE Save a single GraphBLAS matrix to a file.
% GrB.save (C) saves a single GrB or built-in matrix C to a file, with a
% filename of 'C.mat' that matches the matrix name.  If C is an expression, the
% filename 'GrB_Matrix.mat' is used.  A second parameter allows for the
% selection of a different filename, as GrB.save (C, 'myfile.mat').  If A is
% not already a GrB matrix, it is converted to one with GrB(A).
%
% NOTE: As of GraphBLAS v10.4.0, this method is no longer needed in MATLAB;
% just MATLAB load/save methods instead.  Octave cannot load/save the GrB and
% GhB objects, so this method is useful for Octave.  For Octave, save/load
% the serialized blob created by GrB.serialize or GhB.serialize instead.
%
% Example:
%
%   A = magic (4) ;
%   GrB.save (A) ;              % A can be a GrB or built-in matrix
%   clear all
%   A = GrB.load ('A.mat') ;    % A is now a GrB matrix
%
% See also load, save, GrB.load, GrB.serialize, GrB.deserialize.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% determine the default filename
if (nargin < 2)
    filename = inputname (1) ;
    if (isempty (filename))
        % inputname returns an empty string if the input argument C
        % is an expression that has no name
        filename = 'GrB_Matrix' ;
    end
    filename = [filename '.mat'] ;
end

% use the overloaded GrB/saveobj or GhB/saveobj methods to save to the file
save (filename, 'GrB_Matrix_from_GrB_save') ;

% return the chosen filename
if (nargout > 0)
    filename_used = filename ;
end

