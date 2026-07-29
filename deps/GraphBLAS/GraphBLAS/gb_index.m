function [I, whole] = gb_index (I_input)
%GB_INDEX helper function for subsref and subsasgn.  Not user-callable.
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
% struct).  In this case, it is wrapped in a cell, I = { I },
% but kept as 1-based indices (they are later translated to 0-based).
%
% If the input is already a cell array, then it is already in one of the
% above forms.
%
% The subsref and subsasgn methods are passed the string I = ':'.  This is
% converted into I = { }.
%
% If I is a built-in matrix or vector (not a cell array), then it is
% wrapped in a cell array, { I }, to denote A(I).

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

whole = false ;

if (iscell (I_input))

    % The index I_input already appears as a cell, for the usage
    % C ({ }), C ({ I }), C ({start,fini}), or C ({start,inc,fini}).
    len = length (I_input) ;
    if (len > 3)
        error ('GrB:error', 'invalid indexing: usage is A ({start,inc,fini})') ;
    elseif (len == 0)
        % C ({ })
        whole = true ;
    end
    I = I_input ;

elseif (ischar (I_input) && isequal (I_input, ':'))

    % C (:)
    I = { } ;
    whole = true ;

else

    % C (I_input) where I_input is a built-in or GraphBLAS matrix/vector of
    % integer indices, or a GraphBLAS opaque struct
    I = { I_input } ;

end

% replace all GrB matrices with their struct
for k = 1:numel (I)
    if (gb_is_grb (I {k}))
        I {k} = struct (I {k}) ;
    end
end

