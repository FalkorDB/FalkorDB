function [C, I, J] = gb_compact_worker (ghb, A, symmetric)
%GB_COMPACT_WORKER: helper function for gb_compact.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% get the list of non-empty rows and columns
I = gb_entries (1, A, 'row', 'list') ;
J = gb_entries (1, A, 'col', 'list') ;

if (symmetric)
    I = union (I, J) ;
    J = I ;
end

% C = A (I,J)
C = gzb_extract (ghb, A, { I }, { J }) ;

