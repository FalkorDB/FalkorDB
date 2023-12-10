function gbtest100
%GBTEST100 test GrB.ver and GrB.version

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

GrB.MATLAB_vs_GrB ;

fprintf ('v = GrB.ver\n') ;
v = GrB.ver ;
display (v) ;

fprintf ('v = GrB.version\n') ;
v = GrB.version ;
display (v) ;

fprintf ('GrB.ver\n\n') ;
GrB.ver

