function result = GB_isequal_ignore_32 (C1, C2)
%GB_ISEQUAL_IGNORE_32 compare two structs but ignore [phi]_is_32 fields

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

f = { 'p_is_32', 'j_is_32', 'i_is_32' } ;
result = isequal (rmfield (C1, f), rmfield (C2, f)) ;

