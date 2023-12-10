function filename_used = save (C, filename)
%GRB.SAVE Save a single GraphBLAS matrix to a file.
% GrB.save (C) saves a single @GrB or built-in matrix C to a file, with a
% filename of 'C.mat' that matches the matrix name.  If C is an
% expression, the filename 'GrB_Matrix.mat' is used.  A second parameter
% allows for the selection of a different filename, as GrB.save (C,
% 'myfile.mat').  If A is not already a @GrB matrix, it is converted to
% one with GrB(A).
%
% The object or matrix C is saved as a struct containing the opaque
% contents of the GrB object, which is then reconstructed by GrB.load.  A
% matrix saved to a file with GrB.save must be loaded back with GrB.load.
% It cannot be loaded with the built-in load method.
%
% Example:
%
%   A = magic (4) ;
%   GrB.save (A) ;              % A can be a @GrB or built-in matrix
%   clear all
%   A = GrB.load ('A.mat') ;    % A is now a @GrB matrix
%
% See also GrB.load, GrB/struct, GrB.serialize, GrB.deserialize.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% make sure C is a @GrB object
if (~isobject (C))
    C = GrB (C) ;
end

% Extract the opaque contents of C as a struct.  Give it a long and peculiar
% name to help ensure it is only loaded by GrB.load.
GraphBLAS_struct_from_GrB_save = C.opaque ;

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

% save the struct (not the @GrB matrix C) to the file. 
save (filename, 'GraphBLAS_struct_from_GrB_save') ;

% return the chosen filename
if (nargout > 0)
    filename_used = filename ;
end

