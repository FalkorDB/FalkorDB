function R = GB_spec_mask (C, Mask, Z, C_replace, Mask_complement, identity)
%GB_SPEC_MASK mimic of GrB_mask
%
% Computes C<Mask> = Z, in GraphBLAS notation.
%
% Usage:
% C = GB_spec_mask (C, Mask, Z, C_replace, Mask_complement, identity)
%
% C and Z: matrices of the same size.
%
% optional inputs:
% Mask: if empty or not present, Mask = ones (size (C))
% C_replace: set C to zero first. Default is false.
% Mask_complement: use ~Mask instead of Mask. Default is false.
% Mask_struct: handled by GB_spec_mask.
% identity: the additive identity of the semiring.  Default is zero.
%   This is only needed because the GB_spec_* routines operate on dense
%   matrices, and thus they need to know the value of the implicit 'zero'.
%
% This method operates on both plain matrices and on structs with
% matrix, pattern, and class components.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin < 6)
    identity = 0 ;
end
if (nargin < 5)
    Mask_complement = false ;
end
if (nargin < 4)
    C_replace = false ;
end

if (isstruct (C))
    % apply the mask to both the matrix and the pattern
    R.matrix  = GB_spec_mask (C.matrix,  Mask, Z.matrix,  C_replace, ...
        Mask_complement, identity) ;
    R.pattern = GB_spec_mask (C.pattern, Mask, Z.pattern, C_replace, ...
        Mask_complement, false) ;
    R.class = C.class ;
    return
end

% if (~isequal (size (C), size (Z)))
if (~ ((size (C,1) == size (Z,1)) && (size (C,2) == size (Z,2))))
    size (C)
    size (Z)
    error ('C and Z must have the same size') ;
end
if (~isempty (Mask))
    % if (~isequal (size (C), size (Mask)))
    if (~ ((size (C,1) == size (Mask,1)) && (size (C,2) == size (Mask,2))))
        size (C)
        size (Mask)
        error ('C and Mask must have the same size') ;
    end
end

% replace C if requested
if (C_replace)
    C (:,:) = identity ;
end

if (isempty (Mask))
    % in GraphBLAS, this means Mask is NULL;
    % implicitly, Mask = ones (size (C))
    if (~Mask_complement)
        R = Z ;
    else
        % note that Z need never have been computed
        R = C ;
    end
else
    % form the valued mask. For GraphBLAS, this does the
    % right thing and ignores explicit zeros in Mask.
    Mask = (Mask ~= 0) ;
    if (~Mask_complement)
        % R will equal C where Mask is false
        R = C ;
        % overwrite R with Z where Mask is true
        R (Mask) = Z (Mask) ;
    else
        % Mask is complemented
        % R will equal Z where Mask is false
        R = Z ;
        % overwrite R with C where Mask is true
        R (Mask) = C (Mask) ;
    end
end

