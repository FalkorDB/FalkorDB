function [fname, unsigned, bits] = codegen_type (type)
%CODEGEN_TYPE determine function fname, signed or not
% and # bits a C type

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

unsigned = (type (1) == 'u') ;
switch (type)
    case 'bool'
        fname = 'bool' ;
        bits = 8 ;
    case 'int8_t'
        fname = 'int8' ;
        bits = 8 ;
    case 'uint8_t'
        fname = 'uint8' ;
        bits = 8 ;
    case 'int16_t'
        fname = 'int16' ;
        bits = 16 ;
    case 'uint16_t'
        fname = 'uint16' ;
        bits = 16 ;
    case 'int32_t'
        fname = 'int32' ;
        bits = 32 ;
    case 'uint32_t'
        fname = 'uint32' ;
        bits = 32 ;
    case 'int64_t'
        fname = 'int64' ;
        bits = 64 ;
    case 'uint64_t'
        fname = 'uint64' ;
        bits = 64 ;
    case 'float'
        fname = 'fp32' ;
        bits = 32 ;
    case 'double'
        fname = 'fp64' ;
        bits = 64 ;
    case { 'float complex', 'GxB_FC32_t' }
        fname = 'fc32' ;
        bits = 64 ;
    case { 'double complex', 'GxB_FC64_t' }
        fname = 'fc64' ;
        bits = 128 ;
    case 'GB_void'
        fname = 'any' ;
        bits = nan ;
end

