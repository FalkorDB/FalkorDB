function I = gb_index1 (G)
%GB_INDEX1 get one GraphBLAS index for gb_index
% For C=A(I,J), or C(I,J)=A, the indices I and J must be integer lists.
% They can be passed in as GraphBLAS matrices, to subsref and subsasgn.
% This function converts them into into integer lists so that they can be
% handled by the mexFunctions.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

type = gbtype (G) ;
I = gbextractvalues (G) ;

switch (type)

    case { 'double', 'int64', 'uint64' }
        % the mexFunctions handle these three cases themselves

    case { 'single' }
        % the mexFunctions check the indices later, for non-integers
        I = double (I) ;

    case { 'single complex', 'double complex' }
        error ('GrB:error', 'array indices must be integers') ;

    otherwise
        % any other integer must be typecast to double, int64, or uint64.
        % int64 is fine since any errors will be checked later.
        I = int64 (I) ;
end

