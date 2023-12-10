function [I, whole] = gb_index (I)
%GB_INDEX helper function for subsref and subsasgn
% [I, whole] = gb_index (I) converts I into a cell array of built-in
% matrices or vectors containing integer indices, to access A(I).
%
%   I = { }: this denotes A(:), accessing all rows or all columns.
%       In this case, the parameter whole is returned as true.
%
%   I = { list }: denotes A(list)
%
%   I = { start,fini }: denotes A(start:fini), without forming
%       the explicit list start:fini.
%
%   I = { start,inc,fini }: denotes A(start:inc:fini), without forming
%       the explicit list start:inc:fini.
%
% The input I can be a GraphBLAS matrix (as an object or its opaque
% struct).  In this case, it is wrapped in a cell, I = {gb_index1(I)},
% but kept as 1-based indices (they are later translated to 0-based).
%
% If the input is already a cell array, then it is already in one of the
% above forms.  Any member of the cell array that is a GraphBLAS matrix or
% struct is converted into an index list, with gb_index1(I{k}).
%
% The subsref and subsasgn methods are passed the string I = ':'.  This is
% converted into I = { }.
%
% If I is a built-in matrix or vector (not a cell array), then it is
% wrapped in a cell array, { I }, to denote A(I).

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

whole = false ;

if (isobject (I))

    % C (I) where I is a GraphBLAS matrix/vector of integer indices
    I = I.opaque ;
    I = { (gb_index1 (I)) } ;

elseif (isstruct (I))

    % C (I) where I is the opaque struct of a GrB matrix/vector
    I = { (gb_index1 (I)) } ;

elseif (iscell (I))

    % The index I already appears as a cell, for the usage
    % C ({ }), C ({ I }), C ({start,fini}), or C ({start,inc,fini}).
    len = length (I) ;
    if (len > 3)
        error ('GrB:error', 'invalid indexing: usage is A ({start,inc,fini})') ;
    elseif (len == 0)
        % C ({ })
        whole = true ;
    else
        % C ({ I }), C ({start,fini}), or C ({start,inc,fini})
        for k = 1:length(I)
            K = I {k} ;
            if (isobject (K))
                % C ({ ..., K, ... }) where K is a GraphBLAS object
                K = K.opaque ;
            end
            if (isstruct (K))
                % C ({ ..., K, ... }) where I is a GraphBLAS struct
                I {k} = gb_index1 (K) ;
            end
        end
    end

elseif (ischar (I) && isequal (I, ':'))

    % C (:)
    I = { } ;
    whole = true ;

else

    % C (I) where I is a built-in matrix/vector of integer indices
    I = { I } ;

end

