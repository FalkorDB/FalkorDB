function C = accum_mask (C, Mask, accum, T, C_replace, Mask_complement)
%ACCUM_MASK2 a simpler version of GB_spec_accum_mask
% It does not handle typecasting and it assumes the identity value is zero.
% The purpose is for illustration to describe what the accum/mask operation
% does, not for actual testing.  This file appears in the User Guide.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[nrows ncols] = size (C.matrix) ;
Z.matrix  = zeros (nrows, ncols) ;
Z.pattern = false (nrows, ncols) ;

if (isempty (accum))
    % Z = T, no accum operator
    Z.matrix  = T.matrix ;
    Z.pattern = T.pattern ;
else
    % Z = accum (C,T)
    % apply the binary operator to the set intersection
    p = T.pattern & C.pattern ;
    Z.matrix (p) = GB_spec_op (accum, C.matrix (p), T.matrix (p)) ;

    % copy entries in C but not T
    p = C.pattern & ~T.pattern ;
    Z.matrix (p) = C.matrix (p) ;

    % copy entries in T but not C
    p = T.pattern & ~C.pattern ;
    Z.matrix (p) = T.matrix (p) ;

    % pattern is the set union
    Z.pattern = C.pattern | T.pattern ;
end

% apply the mask to the values and pattern
C.matrix  = mask (C.matrix,  Mask, Z.matrix,  C_replace, Mask_complement) ;
C.pattern = mask (C.pattern, Mask, Z.pattern, C_replace, Mask_complement) ;
end

function C = mask (C, Mask, Z, C_replace, Mask_complement) ;
% replace C if requested
if (C_replace)
    C (:,:) = 0 ;       % assume identity is zero
end
if (isempty (Mask))
    % implicitly, Mask = ones (size (C))
    if (~Mask_complement)
        % this is the default
        C = Z ;
    else
        % note that Z need never have been computed
        C = C ;
    end
else
    % apply the mask
    if (~Mask_complement)
        C (Mask) = Z (Mask) ;
    else
        C (~Mask) = Z (~Mask) ;
    end
end
end


