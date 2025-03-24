function test142(tasks)
%TEST142 test GrB_assign for dense matrices

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin < 1)
    tasks = [ ] ;
end

if (isempty (tasks))
    tasks = {
{          '',        'logical',  1}, ... % ( 20,  20)
{     'first',        'logical',  1}, ... % (  3,  23)
{    'second',        'logical',  1}, ... % (  4,  27)
{      'pair',        'logical',  1}, ... % (  1,  28)
{      'plus',        'logical',  1}, ... % (  1,  29)
{     'minus',        'logical',  1}, ... % (  1,  30)
{     'times',        'logical',  1}, ... % (  1,  31)
{       'div',        'logical',  1}, ... % (  2,  33)
{      'iseq',        'logical',  1}, ... % (  1,  34)
{       'pow',        'logical',  1}, ... % (  1,  35)
{      'isgt',        'logical',  1}, ... % (  1,  36)
{      'islt',        'logical',  1}, ... % (  1,  37)
{      'isle',        'logical',  1}, ... % (  1,  38)
{          '',        'logical',  3}, ... % (  3,  41)
{          '',           'int8',  1}, ... % (  2,  43)
{     'first',           'int8',  1}, ... % (  1,  44)
{          '',           'int8',  2}, ... % (  6,  50)
{    'second',           'int8',  2}, ... % (  1,  51)
{      'pair',           'int8',  2}, ... % (  1,  52)
{      'plus',           'int8',  2}, ... % (  1,  53)
{     'minus',           'int8',  2}, ... % (  2,  55)
{    'rminus',           'int8',  2}, ... % (  2,  57)
{     'times',           'int8',  2}, ... % (  2,  59)
{       'div',           'int8',  2}, ... % (  2,  61)
{      'rdiv',           'int8',  2}, ... % (  2,  63)
{      'iseq',           'int8',  2}, ... % (  1,  64)
{      'isne',           'int8',  2}, ... % (  1,  65)
{       'pow',           'int8',  2}, ... % (  2,  67)
{       'min',           'int8',  2}, ... % (  2,  69)
{       'max',           'int8',  2}, ... % (  2,  71)
{      'isgt',           'int8',  2}, ... % (  1,  72)
{      'islt',           'int8',  2}, ... % (  1,  73)
{      'isge',           'int8',  2}, ... % (  2,  75)
{      'isle',           'int8',  2}, ... % (  2,  77)
{        'or',           'int8',  2}, ... % (  2,  79)
{       'and',           'int8',  2}, ... % (  2,  81)
{       'xor',           'int8',  2}, ... % (  2,  83)
{       'bor',           'int8',  2}, ... % (  2,  85)
{      'band',           'int8',  2}, ... % (  2,  87)
{      'bxor',           'int8',  2}, ... % (  2,  89)
{     'bxnor',           'int8',  2}, ... % (  2,  91)
{      'bget',           'int8',  2}, ... % (  2,  93)
{      'bset',           'int8',  2}, ... % (  2,  95)
{      'bclr',           'int8',  2}, ... % (  2,  97)
{    'bshift',           'int8',  2}, ... % (  2,  99)
{          '',          'int16',  2}, ... % (  7, 106)
{    'second',          'int16',  2}, ... % (  1, 107)
{      'pair',          'int16',  2}, ... % (  1, 108)
{      'plus',          'int16',  2}, ... % (  1, 109)
{     'minus',          'int16',  2}, ... % (  2, 111)
{    'rminus',          'int16',  2}, ... % (  2, 113)
{     'times',          'int16',  2}, ... % (  2, 115)
{       'div',          'int16',  2}, ... % (  2, 117)
{      'rdiv',          'int16',  2}, ... % (  2, 119)
{      'iseq',          'int16',  2}, ... % (  1, 120)
{      'isne',          'int16',  2}, ... % (  1, 121)
{       'pow',          'int16',  2}, ... % (  2, 123)
{       'min',          'int16',  2}, ... % (  2, 125)
{       'max',          'int16',  2}, ... % (  2, 127)
{      'isgt',          'int16',  2}, ... % (  1, 128)
{      'islt',          'int16',  2}, ... % (  1, 129)
{      'isge',          'int16',  2}, ... % (  2, 131)
{      'isle',          'int16',  2}, ... % (  2, 133)
{        'or',          'int16',  2}, ... % (  2, 135)
{       'and',          'int16',  2}, ... % (  2, 137)
{       'xor',          'int16',  2}, ... % (  2, 139)
{       'bor',          'int16',  2}, ... % (  2, 141)
{      'band',          'int16',  2}, ... % (  2, 143)
{      'bxor',          'int16',  2}, ... % (  2, 145)
{     'bxnor',          'int16',  2}, ... % (  2, 147)
{      'bget',          'int16',  2}, ... % (  2, 149)
{      'bset',          'int16',  2}, ... % (  2, 151)
{      'bclr',          'int16',  2}, ... % (  2, 153)
{    'bshift',          'int16',  2}, ... % (  1, 154)
{    'bshift',          'int16',  3}, ... % (  1, 155)
{          '',          'int32',  2}, ... % (  6, 161)
{    'second',          'int32',  2}, ... % (  1, 162)
{      'pair',          'int32',  2}, ... % (  1, 163)
{      'plus',          'int32',  2}, ... % (  1, 164)
{     'minus',          'int32',  2}, ... % (  2, 166)
{    'rminus',          'int32',  2}, ... % (  2, 168)
{     'times',          'int32',  2}, ... % (  2, 170)
{       'div',          'int32',  2}, ... % (  2, 172)
{      'rdiv',          'int32',  2}, ... % (  2, 174)
{      'iseq',          'int32',  2}, ... % (  1, 175)
{      'isne',          'int32',  2}, ... % (  1, 176)
{       'pow',          'int32',  2}, ... % (  2, 178)
{       'min',          'int32',  2}, ... % (  2, 180)
{       'max',          'int32',  2}, ... % (  2, 182)
{      'isgt',          'int32',  2}, ... % (  1, 183)
{      'islt',          'int32',  2}, ... % (  1, 184)
{      'isge',          'int32',  2}, ... % (  2, 186)
{      'isle',          'int32',  2}, ... % (  2, 188)
{        'or',          'int32',  2}, ... % (  2, 190)
{       'and',          'int32',  2}, ... % (  2, 192)
{       'xor',          'int32',  2}, ... % (  2, 194)
{       'bor',          'int32',  2}, ... % (  2, 196)
{      'band',          'int32',  2}, ... % (  2, 198)
{      'bxor',          'int32',  2}, ... % (  2, 200)
{     'bxnor',          'int32',  2}, ... % (  2, 202)
{      'bget',          'int32',  2}, ... % (  2, 204)
{      'bset',          'int32',  2}, ... % (  2, 206)
{      'bclr',          'int32',  2}, ... % (  2, 208)
{    'bshift',          'int32',  2}, ... % (  1, 209)
{    'bshift',          'int32',  3}, ... % (  1, 210)
{          '',          'int64',  2}, ... % (  6, 216)
{    'second',          'int64',  2}, ... % (  1, 217)
{      'pair',          'int64',  2}, ... % (  1, 218)
{      'plus',          'int64',  2}, ... % (  1, 219)
{     'minus',          'int64',  2}, ... % (  2, 221)
{    'rminus',          'int64',  2}, ... % (  2, 223)
{     'times',          'int64',  2}, ... % (  2, 225)
{       'div',          'int64',  2}, ... % (  2, 227)
{      'rdiv',          'int64',  2}, ... % (  2, 229)
{      'iseq',          'int64',  2}, ... % (  1, 230)
{      'isne',          'int64',  2}, ... % (  1, 231)
{       'pow',          'int64',  2}, ... % (  2, 233)
{       'min',          'int64',  2}, ... % (  2, 235)
{       'max',          'int64',  2}, ... % (  2, 237)
{      'isgt',          'int64',  2}, ... % (  1, 238)
{      'islt',          'int64',  2}, ... % (  1, 239)
{      'isge',          'int64',  2}, ... % (  2, 241)
{      'isle',          'int64',  2}, ... % (  2, 243)
{        'or',          'int64',  2}, ... % (  2, 245)
{       'and',          'int64',  2}, ... % (  2, 247)
{       'xor',          'int64',  2}, ... % (  2, 249)
{       'bor',          'int64',  2}, ... % (  2, 251)
{      'band',          'int64',  2}, ... % (  2, 253)
{      'bxor',          'int64',  2}, ... % (  2, 255)
{     'bxnor',          'int64',  2}, ... % (  2, 257)
{      'bget',          'int64',  2}, ... % (  2, 259)
{      'bset',          'int64',  2}, ... % (  2, 261)
{      'bclr',          'int64',  2}, ... % (  2, 263)
{    'bshift',          'int64',  2}, ... % (  1, 264)
{    'bshift',          'int64',  3}, ... % (  1, 265)
{          '',          'uint8',  2}, ... % (  6, 271)
{    'second',          'uint8',  2}, ... % (  1, 272)
{      'pair',          'uint8',  2}, ... % (  1, 273)
{      'plus',          'uint8',  2}, ... % (  1, 274)
{     'minus',          'uint8',  2}, ... % (  2, 276)
{    'rminus',          'uint8',  2}, ... % (  2, 278)
{     'times',          'uint8',  2}, ... % (  2, 280)
{       'div',          'uint8',  2}, ... % (  2, 282)
{      'rdiv',          'uint8',  2}, ... % (  2, 284)
{      'iseq',          'uint8',  2}, ... % (  1, 285)
{      'isne',          'uint8',  2}, ... % (  1, 286)
{       'pow',          'uint8',  2}, ... % (  2, 288)
{       'min',          'uint8',  2}, ... % (  2, 290)
{       'max',          'uint8',  2}, ... % (  2, 292)
{      'isgt',          'uint8',  2}, ... % (  1, 293)
{      'islt',          'uint8',  2}, ... % (  1, 294)
{      'isge',          'uint8',  2}, ... % (  2, 296)
{      'isle',          'uint8',  2}, ... % (  2, 298)
{        'or',          'uint8',  2}, ... % (  2, 300)
{       'and',          'uint8',  2}, ... % (  2, 302)
{       'xor',          'uint8',  2}, ... % (  2, 304)
{       'bor',          'uint8',  2}, ... % (  2, 306)
{      'band',          'uint8',  2}, ... % (  2, 308)
{      'bxor',          'uint8',  2}, ... % (  2, 310)
{     'bxnor',          'uint8',  2}, ... % (  2, 312)
{      'bget',          'uint8',  2}, ... % (  2, 314)
{      'bset',          'uint8',  2}, ... % (  2, 316)
{      'bclr',          'uint8',  2}, ... % (  2, 318)
{    'bshift',          'uint8',  2}, ... % (  1, 319)
{    'bshift',          'uint8',  3}, ... % (  1, 320)
{          '',         'uint16',  2}, ... % (  6, 326)
{    'second',         'uint16',  2}, ... % (  1, 327)
{      'pair',         'uint16',  2}, ... % (  1, 328)
{      'plus',         'uint16',  2}, ... % (  1, 329)
{     'minus',         'uint16',  2}, ... % (  2, 331)
{    'rminus',         'uint16',  2}, ... % (  2, 333)
{     'times',         'uint16',  2}, ... % (  2, 335)
{       'div',         'uint16',  2}, ... % (  2, 337)
{      'rdiv',         'uint16',  2}, ... % (  2, 339)
{      'iseq',         'uint16',  2}, ... % (  1, 340)
{      'isne',         'uint16',  2}, ... % (  1, 341)
{       'pow',         'uint16',  2}, ... % (  2, 343)
{       'min',         'uint16',  2}, ... % (  2, 345)
{       'max',         'uint16',  2}, ... % (  2, 347)
{      'isgt',         'uint16',  2}, ... % (  1, 348)
{      'islt',         'uint16',  2}, ... % (  1, 349)
{      'isge',         'uint16',  2}, ... % (  2, 351)
{      'isle',         'uint16',  2}, ... % (  2, 353)
{        'or',         'uint16',  2}, ... % (  2, 355)
{       'and',         'uint16',  2}, ... % (  2, 357)
{       'xor',         'uint16',  2}, ... % (  2, 359)
{       'bor',         'uint16',  2}, ... % (  2, 361)
{      'band',         'uint16',  2}, ... % (  2, 363)
{      'bxor',         'uint16',  2}, ... % (  2, 365)
{     'bxnor',         'uint16',  2}, ... % (  2, 367)
{      'bget',         'uint16',  2}, ... % (  2, 369)
{      'bset',         'uint16',  2}, ... % (  2, 371)
{      'bclr',         'uint16',  2}, ... % (  2, 373)
{    'bshift',         'uint16',  2}, ... % (  1, 374)
{    'bshift',         'uint16',  3}, ... % (  1, 375)
{          '',         'uint32',  2}, ... % (  6, 381)
{    'second',         'uint32',  2}, ... % (  1, 382)
{      'pair',         'uint32',  2}, ... % (  1, 383)
{      'plus',         'uint32',  2}, ... % (  1, 384)
{     'minus',         'uint32',  2}, ... % (  2, 386)
{    'rminus',         'uint32',  2}, ... % (  2, 388)
{     'times',         'uint32',  2}, ... % (  2, 390)
{       'div',         'uint32',  2}, ... % (  2, 392)
{      'rdiv',         'uint32',  2}, ... % (  2, 394)
{      'iseq',         'uint32',  2}, ... % (  1, 395)
{      'isne',         'uint32',  2}, ... % (  1, 396)
{       'pow',         'uint32',  2}, ... % (  2, 398)
{       'min',         'uint32',  2}, ... % (  2, 400)
{       'max',         'uint32',  2}, ... % (  2, 402)
{      'isgt',         'uint32',  2}, ... % (  1, 403)
{      'islt',         'uint32',  2}, ... % (  1, 404)
{      'isge',         'uint32',  2}, ... % (  2, 406)
{      'isle',         'uint32',  2}, ... % (  2, 408)
{        'or',         'uint32',  2}, ... % (  2, 410)
{       'and',         'uint32',  2}, ... % (  2, 412)
{       'xor',         'uint32',  2}, ... % (  2, 414)
{       'bor',         'uint32',  2}, ... % (  2, 416)
{      'band',         'uint32',  2}, ... % (  2, 418)
{      'bxor',         'uint32',  2}, ... % (  2, 420)
{     'bxnor',         'uint32',  2}, ... % (  2, 422)
{      'bget',         'uint32',  2}, ... % (  2, 424)
{      'bset',         'uint32',  2}, ... % (  2, 426)
{      'bclr',         'uint32',  2}, ... % (  2, 428)
{    'bshift',         'uint32',  2}, ... % (  1, 429)
{    'bshift',         'uint32',  3}, ... % (  1, 430)
{          '',         'uint64',  2}, ... % (  6, 436)
{    'second',         'uint64',  2}, ... % (  1, 437)
{      'pair',         'uint64',  2}, ... % (  1, 438)
{      'plus',         'uint64',  2}, ... % (  1, 439)
{     'minus',         'uint64',  2}, ... % (  2, 441)
{    'rminus',         'uint64',  2}, ... % (  2, 443)
{     'times',         'uint64',  2}, ... % (  2, 445)
{       'div',         'uint64',  2}, ... % (  2, 447)
{      'rdiv',         'uint64',  2}, ... % (  2, 449)
{      'iseq',         'uint64',  2}, ... % (  1, 450)
{      'isne',         'uint64',  2}, ... % (  1, 451)
{       'pow',         'uint64',  2}, ... % (  2, 453)
{       'min',         'uint64',  2}, ... % (  2, 455)
{       'max',         'uint64',  2}, ... % (  2, 457)
{      'isgt',         'uint64',  2}, ... % (  1, 458)
{      'islt',         'uint64',  2}, ... % (  1, 459)
{      'isge',         'uint64',  2}, ... % (  2, 461)
{      'isle',         'uint64',  2}, ... % (  2, 463)
{        'or',         'uint64',  2}, ... % (  2, 465)
{       'and',         'uint64',  2}, ... % (  2, 467)
{       'xor',         'uint64',  2}, ... % (  2, 469)
{       'bor',         'uint64',  2}, ... % (  2, 471)
{      'band',         'uint64',  2}, ... % (  2, 473)
{      'bxor',         'uint64',  2}, ... % (  2, 475)
{     'bxnor',         'uint64',  2}, ... % (  2, 477)
{      'bget',         'uint64',  2}, ... % (  2, 479)
{      'bset',         'uint64',  2}, ... % (  2, 481)
{      'bclr',         'uint64',  2}, ... % (  2, 483)
{    'bshift',         'uint64',  2}, ... % (  1, 484)
{    'bshift',         'uint64',  3}, ... % (  1, 485)
{          '',         'single',  2}, ... % (  6, 491)
{    'second',         'single',  2}, ... % (  1, 492)
{      'pair',         'single',  2}, ... % (  1, 493)
{      'plus',         'single',  2}, ... % (  1, 494)
{     'minus',         'single',  2}, ... % (  2, 496)
{    'rminus',         'single',  2}, ... % (  2, 498)
{     'times',         'single',  2}, ... % (  2, 500)
{       'div',         'single',  2}, ... % (  2, 502)
{      'rdiv',         'single',  2}, ... % (  2, 504)
{      'iseq',         'single',  2}, ... % (  1, 505)
{      'isne',         'single',  2}, ... % (  1, 506)
{       'pow',         'single',  2}, ... % (  2, 508)
{       'min',         'single',  2}, ... % (  2, 510)
{       'max',         'single',  2}, ... % (  2, 512)
{      'isgt',         'single',  2}, ... % (  1, 513)
{      'islt',         'single',  2}, ... % (  1, 514)
{      'isge',         'single',  2}, ... % (  2, 516)
{      'isle',         'single',  2}, ... % (  2, 518)
{        'or',         'single',  2}, ... % (  2, 520)
{       'and',         'single',  2}, ... % (  2, 522)
{       'xor',         'single',  2}, ... % (  2, 524)
{     'atan2',         'single',  2}, ... % (  2, 526)
{     'hypot',         'single',  2}, ... % (  2, 528)
{      'fmod',         'single',  2}, ... % (  2, 530)
{ 'remainder',         'single',  2}, ... % (  2, 532)
{     'ldexp',         'single',  2}, ... % (  2, 534)
{  'copysign',         'single',  2}, ... % (  2, 536)
{          '',         'double',  2}, ... % (  4, 540)
{    'second',         'double',  2}, ... % (  1, 541)
{      'pair',         'double',  2}, ... % (  1, 542)
{      'plus',         'double',  2}, ... % (  1, 543)
{     'minus',         'double',  2}, ... % (  2, 545)
{    'rminus',         'double',  2}, ... % (  2, 547)
{     'times',         'double',  2}, ... % (  2, 549)
{       'div',         'double',  2}, ... % (  2, 551)
{      'rdiv',         'double',  2}, ... % (  2, 553)
{      'iseq',         'double',  2}, ... % (  1, 554)
{      'isne',         'double',  2}, ... % (  1, 555)
{       'pow',         'double',  2}, ... % (  2, 557)
{       'min',         'double',  2}, ... % (  2, 559)
{       'max',         'double',  2}, ... % (  2, 561)
{      'isgt',         'double',  2}, ... % (  1, 562)
{      'islt',         'double',  2}, ... % (  1, 563)
{      'isge',         'double',  2}, ... % (  2, 565)
{      'isle',         'double',  2}, ... % (  2, 567)
{        'or',         'double',  2}, ... % (  2, 569)
{       'and',         'double',  2}, ... % (  2, 571)
{       'xor',         'double',  2}, ... % (  2, 573)
{     'atan2',         'double',  2}, ... % (  2, 575)
{     'hypot',         'double',  2}, ... % (  2, 577)
{      'fmod',         'double',  2}, ... % (  2, 579)
{ 'remainder',         'double',  2}, ... % (  2, 581)
{     'ldexp',         'double',  2}, ... % (  2, 583)
{  'copysign',         'double',  2}, ... % (  2, 585)
{          '', 'single complex',  2}, ... % (  6, 591)
{    'second', 'single complex',  2}, ... % (  1, 592)
{      'pair', 'single complex',  2}, ... % (  1, 593)
{      'plus', 'single complex',  2}, ... % (  1, 594)
{     'minus', 'single complex',  2}, ... % (  2, 596)
{    'rminus', 'single complex',  2}, ... % (  2, 598)
{     'times', 'single complex',  2}, ... % (  2, 600)
{       'div', 'single complex',  2}, ... % (  2, 602)
{      'rdiv', 'single complex',  2}, ... % (  2, 604)
{      'iseq', 'single complex',  2}, ... % (  1, 605)
{      'isne', 'single complex',  2}, ... % (  1, 606)
{       'pow', 'single complex',  2}, ... % (  2, 608)
{      'pair', 'single complex',  3}, ... % (  1, 609)
{          '', 'double complex',  2}, ... % (  6, 615)
{    'second', 'double complex',  2}, ... % (  1, 616)
{      'pair', 'double complex',  2}, ... % (  1, 617)
{      'plus', 'double complex',  2}, ... % (  1, 618)
{     'minus', 'double complex',  2}, ... % (  2, 620)
{    'rminus', 'double complex',  2}, ... % (  2, 622)
{     'times', 'double complex',  2}, ... % (  2, 624)
{       'div', 'double complex',  2}, ... % (  2, 626)
{      'rdiv', 'double complex',  2}, ... % (  2, 628)
{      'iseq', 'double complex',  2}, ... % (  1, 629)
{      'isne', 'double complex',  2}, ... % (  1, 630)
{       'pow', 'double complex',  2}, ... % (  2, 632)
{      'pair', 'double complex',  3}, ... % (  1, 633)
    } ;
end

track_coverage = false ;
if (track_coverage)
    global GraphBLAS_grbcov
    track_coverage = ~isempty (GraphBLAS_grbcov) ;
    clast = sum (GraphBLAS_grbcov > 0) ;
    cfirst = clast ;
end

% [binops, ~, ~, types, ~, ~] = GB_spec_opsall ;
% binops = binops.all ;
% types = types.all ;

fprintf ('test142 ------------ GrB_assign with dense matrices\n') ;

m = 10 ;
n = 12 ;

% create the test matrices
rng ('default') ;
M = sprand (m, n, 0.5) ;
Amat2 = sparse (2 * rand (m,n)) ;
Bmat2 = sparse (2 * sprand (m,n, 0.5)) ;
Cmat2 = sparse (2 * rand (m,n)) ;

Amat = 50 * Amat2 ;
Bmat = 50 * Bmat2 ;
Cmat = 50 * Cmat2 ;

Smat = sparse (m,n) ;
Xmat = sparse (pi) ;
desc.mask = 'structural' ;
drep.outp = 'replace' ;

A.matrix = Amat ; A.class = 'see below' ;
B.matrix = Bmat ; B.class = 'see below' ;
C.matrix = Cmat ; C.class = 'see below' ;
S.matrix = Smat ; S.class = 'see below' ;
X.matrix = Xmat ; X.class = 'see below' ;
Bmask = logical (Bmat) ;
A.sparsity = 8 ;
C.sparsity = 8 ;
X.sparsity = 8 ;

for kk = 1:length(tasks)
    task = tasks {kk} ;
    binop = task {1} ;
    type = task {2} ;
    k3 = task {3} ;

% end
% for k1 = 1:length (types)
%     type = types {k1}  ;
    % fprintf ('%s ', type) ;

    A.class = type ;
    id = test_cast (0, type) ;
    A_iso = A ;
    A_iso.iso = true ;

%   for k3 = 1:3

    if (k3 == 1)
        X.class = type ;
        B.class = type ;
        C.class = 'logical' ;
        S.class = 'logical' ;
    elseif (k3 == 2)
        X.class = type ;
        B.class = type ;
        C.class = type ;
        S.class = type ;
    else
        X.class = 'int8' ;
        B.class = 'int8' ;
        C.class = type ;
        S.class = type ;
    end

%   binop = [ ] ;
    if (isempty (binop))

        %---------------------------------------
        % C<M> = A where A is dense
        %---------------------------------------

        C0 = GB_spec_assign (C, M, [ ], A, [ ], [ ], [ ], false) ;
        C1 = GB_mex_assign  (C, M, [ ], A, [ ], [ ], [ ]) ;
        GB_spec_compare (C0, C1) ;

        %---------------------------------------
        % C<M> = B where B is sparse
        %---------------------------------------

        C0 = GB_spec_assign (C, M, [ ], B, [ ], [ ], [ ], false) ;
        C1 = GB_mex_assign  (C, M, [ ], B, [ ], [ ], [ ]) ;
        GB_spec_compare (C0, C1) ;

        %---------------------------------------
        % C<M> = A where A is dense and C starts empty
        %---------------------------------------

        C0 = GB_spec_assign (S, M, [ ], A, [ ], [ ], desc, false) ;
        C1 = GB_mex_assign  (S, M, [ ], A, [ ], [ ], desc) ;
        GB_spec_compare (C0, C1) ;

        %---------------------------------------
        % C<M> = A where A is iso full and C starts empty
        %---------------------------------------

        C0 = GB_spec_assign (S, M, [ ], A_iso, [ ], [ ], desc, false) ;
        C1 = GB_mex_assign  (S, M, [ ], A_iso, [ ], [ ], desc) ;
        GB_spec_compare (C0, C1) ;

        %---------------------------------------
        % C<B> = B where B is sparse
        %---------------------------------------

        C0 = GB_spec_assign (C, Bmask, [ ], B, [ ], [ ], desc, false) ;
        C1 = GB_mex_assign_alias_mask (C, B, desc) ;
        GB_spec_compare (C0, C1) ;

        %---------------------------------------
        % C<M> = x where C is dense
        %---------------------------------------

        C0 = GB_spec_assign (C, M, [ ], X, [ ], [ ], [ ], true) ;
        C1 = GB_mex_assign  (C, M, [ ], X, [ ], [ ], [ ]) ;
        GB_spec_compare (C0, C1) ;

        %---------------------------------------
        % C<M> = x where C is dense
        %---------------------------------------

        C0 = GB_spec_assign (C, M, [ ], X, [ ], [ ], desc, true) ;
        C1 = GB_mex_assign  (C, M, [ ], X, [ ], [ ], desc) ;
        GB_spec_compare (C0, C1) ;

        %---------------------------------------
        % C<M,struct> = x
        %---------------------------------------

        C0 = GB_spec_assign (S, M, [ ], X, [ ], [ ], desc, true) ;
        C1 = GB_mex_assign  (S, M, [ ], X, [ ], [ ], desc) ;
        GB_spec_compare (C0, C1) ;

        %---------------------------------------
        % C = x
        %---------------------------------------

        C0 = GB_spec_assign (S, [ ], [ ], X, [ ], [ ], [ ], true) ;
        C1 = GB_mex_assign  (S, [ ], [ ], X, [ ], [ ], [ ]) ;
        GB_spec_compare (C0, C1) ;

            if (track_coverage)
                c = sum (GraphBLAS_grbcov > 0) ;
                d = c - clast ;
                if (d > 0)
                    tt = sprintf ('''%s''', type) ;
                    oo = sprintf ('''%s''', '') ;
                    fprintf ('{%12s, %16s, %2d},', oo, tt, k3) ;
                    fprintf (' ... %% (%3d, %3d)\n', d, c-cfirst) ;
                end
                clast = c ;
            else
                fprintf ('.') ;
            end
    end

        %---------------------------------------
        % with accum operators
        %---------------------------------------

%       for k2 = 1:length(binops)
%           binop = binops {k2}  ;

    if (~isempty (binop))

        tol = [ ] ;
        switch (binop)
            case { 'pow', 'atan2', 'hypot', 'remainder' }
                A.matrix = Amat2 ;
                B.matrix = Bmat2 ;
                C.matrix = Cmat2 ;
                if (test_contains (type, 'single'))
                    tol = 1e-5 ;
                elseif (test_contains (type, 'double'))
                    tol = 1e-12 ;
                end
            otherwise
                A.matrix = Amat ;
                B.matrix = Bmat ;
                C.matrix = Cmat ;
        end

        accum.opname = binop ;
        accum.optype = type ;

        try
            GB_spec_operator (accum) ;
        catch
            continue
        end

        if (GB_spec_is_positional (accum))
            continue ;
        end

        %---------------------------------------
        % C += A where A is dense
        %---------------------------------------

        C0 = GB_spec_assign (C, [ ], accum, A, [ ], [ ], [ ], false) ;
        C1 = GB_mex_assign  (C, [ ], accum, A, [ ], [ ], [ ]) ;
        GB_spec_compare (C0, C1, id, tol) ;

        %---------------------------------------
        % C += B where B is sparse
        %---------------------------------------

        C0 = GB_spec_assign (C, [ ], accum, B, [ ], [ ], [ ], false) ;
        C1 = GB_mex_assign  (C, [ ], accum, B, [ ], [ ], [ ]) ;
        GB_spec_compare (C0, C1, id, tol) ;

        %---------------------------------------
        % C += x
        %---------------------------------------

        C0 = GB_spec_assign (C, [ ], accum, X, [ ], [ ], [ ], true) ;
        C1 = GB_mex_assign  (C, [ ], accum, X, [ ], [ ], [ ]) ;
        GB_spec_compare (C0, C1, id, tol) ;

        %---------------------------------------
        % C<replace> += x
        %---------------------------------------

        C0 = GB_spec_assign (C, [ ], accum, X, [ ], [ ], drep, true) ;
        C1 = GB_mex_subassign  (C, [ ], accum, X, [ ], [ ], drep) ;
        GB_spec_compare (C0, C1, id, tol) ;

        if (track_coverage)
            c = sum (GraphBLAS_grbcov > 0) ;
            d = c - clast ;
            if (d > 0)
                tt = sprintf ('''%s''', type) ;
                oo = sprintf ('''%s''', binop) ;
                fprintf ('{%12s, %16s, %2d},', oo, tt, k3) ;
                fprintf (' ... %% (%3d, %3d)\n', d, c-cfirst) ;
            end
            clast = c ;
        else
            fprintf ('.') ;
        end

    end
end

fprintf ('\ntest142: all tests passed\n') ;

