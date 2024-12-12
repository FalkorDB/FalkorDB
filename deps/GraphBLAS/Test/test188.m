function test188(tasks)
%TEST188 test concat

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% NOTE: the test coverage requires the JIT cache to be empty first,
% for full coverage.

fprintf ('test188 ----------- C = concat (Tiles)\n') ;

if (nargin < 1)
    tasks = [ ] ;
end

if (isempty (tasks))
    tasks = {
    {       'logical',        'logical', 1, 0, 1, 0, 0}, ... % (2, 2)
    {       'logical',         'double', 1, 0, 1, 0, 0}, ... % (4, 6)
    {       'logical',         'double', 1, 0, 1, 0, 1}, ... % (3, 9)
    {          'int8',         'double', 1, 0, 1, 0, 1}, ... % (1, 10)
    {         'int16',         'double', 1, 0, 1, 0, 1}, ... % (2, 12)
    {         'int32',         'double', 1, 0, 1, 0, 1}, ... % (2, 14)
    {         'int64',         'double', 1, 0, 1, 0, 1}, ... % (2, 16)
    {         'uint8',         'double', 1, 0, 1, 0, 1}, ... % (1, 17)
    {        'uint16',         'double', 1, 0, 1, 0, 1}, ... % (1, 18)
    {        'uint32',         'double', 1, 0, 1, 0, 1}, ... % (1, 19)
    {        'uint64',         'double', 1, 0, 1, 0, 1}, ... % (1, 20)
    {'single complex',         'double', 1, 0, 1, 0, 1}, ... % (1, 21)
    {'double complex',         'double', 1, 0, 1, 0, 1}, ... % (2, 23)
    {       'logical',         'double', 1, 0, 1, 1, 0}, ... % (1, 24)
    {       'logical',         'double', 1, 0, 4, 0, 0}, ... % (1, 25)
    {          'int8',        'logical', 2, 0, 4, 0, 1}, ... % (1, 26)
    {         'int16',        'logical', 2, 0, 4, 0, 1}, ... % (1, 27)
    {         'int32',        'logical', 2, 0, 4, 0, 1}, ... % (1, 28)
    {         'int64',        'logical', 2, 0, 4, 0, 1}, ... % (1, 29)
    {         'uint8',        'logical', 2, 0, 4, 0, 1}, ... % (1, 30)
    {        'uint16',        'logical', 2, 0, 4, 0, 1}, ... % (1, 31)
    {        'uint32',        'logical', 2, 0, 4, 0, 1}, ... % (1, 32)
    {        'uint64',        'logical', 2, 0, 4, 0, 1}, ... % (1, 33)
    {        'single',        'logical', 2, 0, 4, 0, 1}, ... % (1, 34)
    {        'double',        'logical', 2, 0, 4, 0, 1}, ... % (1, 35)
    {'single complex',        'logical', 2, 0, 4, 0, 1}, ... % (1, 36)
    {'double complex',        'logical', 2, 0, 4, 0, 1}, ... % (1, 37)
    {       'logical',           'int8', 2, 0, 4, 0, 1}, ... % (1, 38)
    {         'int16',           'int8', 2, 0, 4, 0, 1}, ... % (1, 39)
    {         'int32',           'int8', 2, 0, 4, 0, 1}, ... % (1, 40)
    {         'int64',           'int8', 2, 0, 4, 0, 1}, ... % (1, 41)
    {         'uint8',           'int8', 2, 0, 4, 0, 1}, ... % (1, 42)
    {        'uint16',           'int8', 2, 0, 4, 0, 1}, ... % (1, 43)
    {        'uint32',           'int8', 2, 0, 4, 0, 1}, ... % (1, 44)
    {        'uint64',           'int8', 2, 0, 4, 0, 1}, ... % (1, 45)
    {        'single',           'int8', 2, 0, 4, 0, 1}, ... % (1, 46)
    {        'double',           'int8', 2, 0, 4, 0, 1}, ... % (1, 47)
    {'single complex',           'int8', 2, 0, 4, 0, 1}, ... % (1, 48)
    {'double complex',           'int8', 2, 0, 4, 0, 1}, ... % (1, 49)
    {       'logical',          'int16', 2, 0, 4, 0, 1}, ... % (1, 50)
    {          'int8',          'int16', 2, 0, 4, 0, 1}, ... % (1, 51)
    {         'int32',          'int16', 2, 0, 4, 0, 1}, ... % (1, 52)
    {         'int64',          'int16', 2, 0, 4, 0, 1}, ... % (1, 53)
    {         'uint8',          'int16', 2, 0, 4, 0, 1}, ... % (1, 54)
    {        'uint16',          'int16', 2, 0, 4, 0, 1}, ... % (1, 55)
    {        'uint32',          'int16', 2, 0, 4, 0, 1}, ... % (1, 56)
    {        'uint64',          'int16', 2, 0, 4, 0, 1}, ... % (1, 57)
    {        'single',          'int16', 2, 0, 4, 0, 1}, ... % (1, 58)
    {        'double',          'int16', 2, 0, 4, 0, 1}, ... % (1, 59)
    {'single complex',          'int16', 2, 0, 4, 0, 1}, ... % (1, 60)
    {'double complex',          'int16', 2, 0, 4, 0, 1}, ... % (1, 61)
    {       'logical',          'int32', 2, 0, 4, 0, 1}, ... % (1, 62)
    {          'int8',          'int32', 2, 0, 4, 0, 1}, ... % (1, 63)
    {         'int16',          'int32', 2, 0, 4, 0, 1}, ... % (1, 64)
    {         'int64',          'int32', 2, 0, 4, 0, 1}, ... % (1, 65)
    {         'uint8',          'int32', 2, 0, 4, 0, 1}, ... % (1, 66)
    {        'uint16',          'int32', 2, 0, 4, 0, 1}, ... % (1, 67)
    {        'uint32',          'int32', 2, 0, 4, 0, 1}, ... % (1, 68)
    {        'uint64',          'int32', 2, 0, 4, 0, 1}, ... % (1, 69)
    {        'single',          'int32', 2, 0, 4, 0, 1}, ... % (1, 70)
    {        'double',          'int32', 2, 0, 4, 0, 1}, ... % (1, 71)
    {'single complex',          'int32', 2, 0, 4, 0, 1}, ... % (1, 72)
    {'double complex',          'int32', 2, 0, 4, 0, 1}, ... % (1, 73)
    {       'logical',          'int64', 2, 0, 4, 0, 1}, ... % (1, 74)
    {          'int8',          'int64', 2, 0, 4, 0, 1}, ... % (1, 75)
    {         'int16',          'int64', 2, 0, 4, 0, 1}, ... % (1, 76)
    {         'int32',          'int64', 2, 0, 4, 0, 1}, ... % (1, 77)
    {         'uint8',          'int64', 2, 0, 4, 0, 1}, ... % (1, 78)
    {        'uint16',          'int64', 2, 0, 4, 0, 1}, ... % (1, 79)
    {        'uint32',          'int64', 2, 0, 4, 0, 1}, ... % (1, 80)
    {        'uint64',          'int64', 2, 0, 4, 0, 1}, ... % (1, 81)
    {        'single',          'int64', 2, 0, 4, 0, 1}, ... % (1, 82)
    {        'double',          'int64', 2, 0, 4, 0, 1}, ... % (1, 83)
    {'single complex',          'int64', 2, 0, 4, 0, 1}, ... % (1, 84)
    {'double complex',          'int64', 2, 0, 4, 0, 1}, ... % (1, 85)
    {       'logical',          'uint8', 2, 0, 4, 0, 1}, ... % (1, 86)
    {          'int8',          'uint8', 2, 0, 4, 0, 1}, ... % (1, 87)
    {         'int16',          'uint8', 2, 0, 4, 0, 1}, ... % (1, 88)
    {         'int32',          'uint8', 2, 0, 4, 0, 1}, ... % (1, 89)
    {         'int64',          'uint8', 2, 0, 4, 0, 1}, ... % (1, 90)
    {        'uint16',          'uint8', 2, 0, 4, 0, 1}, ... % (1, 91)
    {        'uint32',          'uint8', 2, 0, 4, 0, 1}, ... % (1, 92)
    {        'uint64',          'uint8', 2, 0, 4, 0, 1}, ... % (1, 93)
    {        'single',          'uint8', 2, 0, 4, 0, 1}, ... % (1, 94)
    {        'double',          'uint8', 2, 0, 4, 0, 1}, ... % (1, 95)
    {'single complex',          'uint8', 2, 0, 4, 0, 1}, ... % (1, 96)
    {'double complex',          'uint8', 2, 0, 4, 0, 1}, ... % (1, 97)
    {       'logical',         'uint16', 2, 0, 4, 0, 1}, ... % (1, 98)
    {          'int8',         'uint16', 2, 0, 4, 0, 1}, ... % (1, 99)
    {         'int16',         'uint16', 2, 0, 4, 0, 1}, ... % (1, 100)
    {         'int32',         'uint16', 2, 0, 4, 0, 1}, ... % (1, 101)
    {         'int64',         'uint16', 2, 0, 4, 0, 1}, ... % (1, 102)
    {         'uint8',         'uint16', 2, 0, 4, 0, 1}, ... % (1, 103)
    {        'uint32',         'uint16', 2, 0, 4, 0, 1}, ... % (1, 104)
    {        'uint64',         'uint16', 2, 0, 4, 0, 1}, ... % (1, 105)
    {        'single',         'uint16', 2, 0, 4, 0, 1}, ... % (1, 106)
    {        'double',         'uint16', 2, 0, 4, 0, 1}, ... % (1, 107)
    {'single complex',         'uint16', 2, 0, 4, 0, 1}, ... % (1, 108)
    {'double complex',         'uint16', 2, 0, 4, 0, 1}, ... % (1, 109)
    {       'logical',         'uint32', 2, 0, 4, 0, 1}, ... % (1, 110)
    {          'int8',         'uint32', 2, 0, 4, 0, 1}, ... % (1, 111)
    {         'int16',         'uint32', 2, 0, 4, 0, 1}, ... % (1, 112)
    {         'int32',         'uint32', 2, 0, 4, 0, 1}, ... % (1, 113)
    {         'int64',         'uint32', 2, 0, 4, 0, 1}, ... % (1, 114)
    {         'uint8',         'uint32', 2, 0, 4, 0, 1}, ... % (1, 115)
    {        'uint16',         'uint32', 2, 0, 4, 0, 1}, ... % (1, 116)
    {        'uint64',         'uint32', 2, 0, 4, 0, 1}, ... % (1, 117)
    {        'single',         'uint32', 2, 0, 4, 0, 1}, ... % (1, 118)
    {        'double',         'uint32', 2, 0, 4, 0, 1}, ... % (1, 119)
    {'single complex',         'uint32', 2, 0, 4, 0, 1}, ... % (1, 120)
    {'double complex',         'uint32', 2, 0, 4, 0, 1}, ... % (1, 121)
    {       'logical',         'uint64', 2, 0, 4, 0, 1}, ... % (1, 122)
    {          'int8',         'uint64', 2, 0, 4, 0, 1}, ... % (1, 123)
    {         'int16',         'uint64', 2, 0, 4, 0, 1}, ... % (1, 124)
    {         'int32',         'uint64', 2, 0, 4, 0, 1}, ... % (1, 125)
    {         'int64',         'uint64', 2, 0, 4, 0, 1}, ... % (1, 126)
    {         'uint8',         'uint64', 2, 0, 4, 0, 1}, ... % (1, 127)
    {        'uint16',         'uint64', 2, 0, 4, 0, 1}, ... % (1, 128)
    {        'uint32',         'uint64', 2, 0, 4, 0, 1}, ... % (1, 129)
    {        'single',         'uint64', 2, 0, 4, 0, 1}, ... % (1, 130)
    {        'double',         'uint64', 2, 0, 4, 0, 1}, ... % (1, 131)
    {'single complex',         'uint64', 2, 0, 4, 0, 1}, ... % (1, 132)
    {'double complex',         'uint64', 2, 0, 4, 0, 1}, ... % (1, 133)
    {          'int8',         'single', 2, 0, 1, 0, 0}, ... % (1, 134)
    {         'int32',         'single', 2, 0, 1, 0, 0}, ... % (1, 135)
    {         'int64',         'single', 2, 0, 1, 0, 0}, ... % (1, 136)
    {         'uint8',         'single', 2, 0, 1, 0, 0}, ... % (1, 137)
    {        'uint32',         'single', 2, 0, 1, 0, 0}, ... % (1, 138)
    {        'uint64',         'single', 2, 0, 1, 0, 0}, ... % (1, 139)
    {       'logical',         'single', 2, 0, 4, 0, 1}, ... % (1, 140)
    {          'int8',         'single', 2, 0, 4, 0, 1}, ... % (1, 141)
    {         'int16',         'single', 2, 0, 4, 0, 1}, ... % (1, 142)
    {         'int32',         'single', 2, 0, 4, 0, 1}, ... % (1, 143)
    {         'int64',         'single', 2, 0, 4, 0, 1}, ... % (1, 144)
    {         'uint8',         'single', 2, 0, 4, 0, 1}, ... % (1, 145)
    {        'uint16',         'single', 2, 0, 4, 0, 1}, ... % (1, 146)
    {        'uint32',         'single', 2, 0, 4, 0, 1}, ... % (1, 147)
    {        'uint64',         'single', 2, 0, 4, 0, 1}, ... % (1, 148)
    {'single complex',         'single', 2, 0, 4, 0, 1}, ... % (1, 149)
    {'double complex',         'single', 2, 0, 4, 0, 1}, ... % (1, 150)
    {       'logical', 'single complex', 2, 0, 1, 0, 0}, ... % (1, 151)
    {       'logical', 'single complex', 2, 0, 1, 0, 1}, ... % (2, 153)
    {          'int8', 'single complex', 2, 0, 1, 0, 0}, ... % (1, 154)
    {          'int8', 'single complex', 2, 0, 1, 0, 1}, ... % (1, 155)
    {         'int16', 'single complex', 2, 0, 1, 0, 1}, ... % (1, 156)
    {         'int32', 'single complex', 2, 0, 1, 0, 0}, ... % (1, 157)
    {         'int32', 'single complex', 2, 0, 1, 0, 1}, ... % (1, 158)
    {         'int64', 'single complex', 2, 0, 1, 0, 0}, ... % (1, 159)
    {         'int64', 'single complex', 2, 0, 1, 0, 1}, ... % (1, 160)
    {         'uint8', 'single complex', 2, 0, 1, 0, 0}, ... % (1, 161)
    {         'uint8', 'single complex', 2, 0, 1, 0, 1}, ... % (1, 162)
    {        'uint16', 'single complex', 2, 0, 1, 0, 1}, ... % (1, 163)
    {        'uint32', 'single complex', 2, 0, 1, 0, 0}, ... % (1, 164)
    {        'uint32', 'single complex', 2, 0, 1, 0, 1}, ... % (1, 165)
    {        'uint64', 'single complex', 2, 0, 1, 0, 0}, ... % (1, 166)
    {        'uint64', 'single complex', 2, 0, 1, 0, 1}, ... % (1, 167)
    {        'single', 'single complex', 2, 0, 1, 0, 1}, ... % (1, 168)
    {        'double', 'single complex', 2, 0, 1, 0, 1}, ... % (1, 169)
    {'double complex', 'single complex', 2, 0, 1, 0, 1}, ... % (1, 170)
    {       'logical', 'double complex', 2, 0, 1, 0, 1}, ... % (1, 171)
    {          'int8', 'double complex', 2, 0, 1, 0, 0}, ... % (1, 172)
    {          'int8', 'double complex', 2, 0, 1, 0, 1}, ... % (1, 173)
    {         'int16', 'double complex', 2, 0, 1, 0, 1}, ... % (1, 174)
    {         'int32', 'double complex', 2, 0, 1, 0, 1}, ... % (1, 175)
    {         'int64', 'double complex', 2, 0, 1, 0, 0}, ... % (1, 176)
    {         'int64', 'double complex', 2, 0, 1, 0, 1}, ... % (1, 177)
    {         'uint8', 'double complex', 2, 0, 1, 0, 0}, ... % (1, 178)
    {         'uint8', 'double complex', 2, 0, 1, 0, 1}, ... % (1, 179)
    {        'uint16', 'double complex', 2, 0, 1, 0, 1}, ... % (1, 180)
    {        'uint32', 'double complex', 2, 0, 1, 0, 0}, ... % (1, 181)
    {        'uint32', 'double complex', 2, 0, 1, 0, 1}, ... % (1, 182)
    {        'uint64', 'double complex', 2, 0, 1, 0, 0}, ... % (1, 183)
    {        'uint64', 'double complex', 2, 0, 1, 0, 1}, ... % (1, 184)
    {        'single', 'double complex', 2, 0, 1, 0, 1}, ... % (1, 185)
    {        'double', 'double complex', 2, 0, 1, 0, 1}, ... % (1, 186)
    {'single complex', 'double complex', 2, 0, 1, 0, 1}, ... % (1, 187)
    {       'logical',         'double', 3, 0, 1, 0, 0}, ... % (2, 189)
    {       'logical',         'double', 3, 0, 1, 0, 1}, ... % (4, 193)
    {         'int16',         'double', 3, 0, 1, 0, 1}, ... % (1, 194)
    {         'int32',         'double', 3, 0, 1, 0, 1}, ... % (1, 195)
    {         'int64',         'double', 3, 0, 1, 0, 1}, ... % (1, 196)
    {'double complex',         'double', 3, 0, 1, 0, 1}, ... % (1, 197)
    {       'logical',        'logical', 4, 0, 8, 0, 0}, ... % (4, 201)
    {       'logical',        'logical', 4, 0, 8, 0, 1}, ... % (1, 202)
    {         'int16',        'logical', 4, 0, 8, 0, 1}, ... % (1, 203)
    {         'int32',        'logical', 4, 0, 8, 0, 1}, ... % (1, 204)
    {         'int64',        'logical', 4, 0, 8, 0, 1}, ... % (1, 205)
    {'double complex',        'logical', 4, 0, 8, 0, 1}} ;   % (1, 206)
end

track_coverage = false ;
if (track_coverage)
    global GraphBLAS_grbcov
    track_coverage = ~isempty (GraphBLAS_grbcov) ;
    clast = sum (GraphBLAS_grbcov > 0) ;
    cfirst = clast ;
end

rng ('default') ;

n1 = 20 ;
n2 = 4 ;
k1_last = 0 ;
atype_last = '' ;

densities = [1e-4 0.01 0.2 inf] ;

for ktask = 1:length (tasks)
    task = tasks {ktask} ;
    ctype = task {1} ;
    atype = task {2} ;
    k1 = task {3} ;
    iso = task {4} ;
    sparsity_control = task {5} ;
    is_csc = task {6} ;
    fmt = task {7} ;

    density = densities (k1) ;

    if (k1 ~= k1_last || ~isequal (atype, atype_last))
        % create the test matrices; note the tasks are sorted by density and
        % atype.  If new tasks are added, this sorting order should be
        % preserved and no new matrices should be created in the list of
        % existing tasks.  Append new matrices at the end.  Otherwise, the
        % matrices for the existing tasks will differ, and the coverage might
        % change.
        A1 = GB_spec_random (n1, n1, density, 128, atype) ;
        A2 = GB_spec_random (n1, n2, density, 128, atype) ;
        A3 = GB_spec_random (n2, n1, density, 128, atype) ;
        if (isequal (atype, 'double'))
            A4 = GB_spec_random (n2, n2, inf, 128, atype) ;
        else
            A4 = GB_spec_random (n2, n2, density, 128, atype) ;
        end
    end

    k1_last = k1 ;
    atype_last = atype ;

    % test iso case
    A1.iso = iso ;
    A2.iso = iso ;
    A3.iso = iso ;
    A4.iso = iso ;

    A1.sparsity = sparsity_control ;
    A2.sparsity = sparsity_control ;
    A3.sparsity = sparsity_control ;
    if (isequal (atype, 'double'))
        A4.sparsity = 8 ;
    else
        A4.sparsity = sparsity_control ;
    end

    A1.is_csc = is_csc ;
    A2.is_csc = is_csc ;
    A3.is_csc = is_csc ;
    A4.is_csc = is_csc ;
    Tiles = cell (2,2) ;
    Tiles {1,1} = A1 ;
    Tiles {1,2} = A2 ;
    Tiles {2,1} = A3 ;
    Tiles {2,2} = A4 ;

    C1 = GB_mex_concat  (Tiles, ctype, fmt) ;
    C2 = GB_spec_concat (Tiles, ctype) ;
    GB_spec_compare (C1, C2) ;

    if (track_coverage)
        c = sum (GraphBLAS_grbcov > 0) ;
        d = c - clast ;
        if (d > 0)
            cc = sprintf ('''%s''', ctype) ;
            aa = sprintf ('''%s''', atype) ;
            fprintf (...
            '{%16s, %16s, %d, %d, %d, %d, %d},', ...
            cc, aa, k1, iso, sparsity_control, ...
            is_csc, fmt) ;
            fprintf (' ... %% (%d, %d)\n', ...
                d, c-cfirst) ;
        end
        clast = c ;
    else
        fprintf ('.') ;
    end
end

fprintf ('\n') ;
fprintf ('test188: all tests passed\n') ;

