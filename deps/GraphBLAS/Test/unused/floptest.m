function floptest (M, Mask_complement, A, B, flops1)
%FLOPTEST compare flopcount with GB_mex_mxm_flops
% floptest (M, Mask_complement, A, B, flops1)
%
% compares the results of
% flops1 = flopcount (M, Mask_complement, A, B)
% with
% flops2 = GB_mex_mxm_flops (M, Mask_complement, A, B)
%
% However, flopcount(M,Mask_complement,A,B) can only be computed when M, A, B
% are all built-in sparse matrices, not structs.  If the matrices are
% hypersparse, flops1 has length B->nvec+1, not size(B,2).  In this case,
% only the total flop count is checked.  In that case, flops1 is a scalar.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[flops2 mwork] = GB_mex_mxm_flops (M, Mask_complement, A, B) ;
total_flops = flops2 (end) ;

if (isscalar (flops1))
    % flops1 is just the total flop count
    assert (isequal (flops1, total_flops)) ;
else
    % flops1 is the cumulative sum
    assert (isequal (flops1, flops2)) ;
end

