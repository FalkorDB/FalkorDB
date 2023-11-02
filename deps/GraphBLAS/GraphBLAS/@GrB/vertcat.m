function C = vertcat (varargin)
%VERTCAT vertical concatenation.
% [A ; B] is the vertical concatenation of A and B.
% Multiple matrices may be concatenated, as [A ; B ; C ; ...].
% If the matrices have different types, the type is determined
% according to the rules in GrB.optype.
%
% See also GrB/horzcat, GrB/cat, GrB.cell2mat, GrB/mat2cell, GrB/num2cell.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% get the input matrices
nmatrices = length (varargin) ;
for k = 1:nmatrices
    Tile = varargin {k} ;
    if (isobject (Tile))
        varargin {k} = Tile.opaque ;
    end
end

% concatenate the matrices
C = GrB (gbcat (varargin')) ;

