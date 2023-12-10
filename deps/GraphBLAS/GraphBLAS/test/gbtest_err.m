function [err, errnan] = gbtest_err (A, B)
%GBTEST_ERR compare two matrices
%
% err = gbtest_err (A, B)
%
% Returns the norm (A-B,1), ignoring inf's and nan's.
% Also tests the result of isinf and isnan for A and B.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

err = 0 ; %#ok<*NASGU>
errnan = false ;

X = isnan (A) ;
Y = isnan (B) ;
if (~gbtest_eq (X, Y))
    errnan = true ;
end
if (nnz (X) > 0)
    A (X) = 0 ;
end
if (nnz (Y) > 0)
    B (Y) = 0 ;
end

X = isinf (A) ;
Y = isinf (B) ;
if (~gbtest_eq (X, Y))
    errnan = true ;
end
if (nnz (X) > 0)
    A (X) = 0 ;
end
if (nnz (Y) > 0)
    B (Y) = 0 ;
end

A (~isfinite (A)) = 0 ;
B (~isfinite (B)) = 0 ;
err = GrB.normdiff (A, B, 1) ;

