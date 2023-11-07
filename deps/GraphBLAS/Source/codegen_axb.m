function codegen_axb
%CODEGEN_AXB create all C=A*B functions for all semirings
%
% This function creates all files of the form GB_AxB__*.[ch], including all
% built-in semirings (GB_AxB__*.c) and two include files,
% Source/GB_AxB__include1.h and FactoryKernels/GB_AxB__include2.h.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% The ANY operator is not used as a multiplicative operator in the generated
% functions.  It can be used as the multiplicative op in a semiring, but is
% renamed to SECOND before calling the generated function.

fprintf ('\nsemirings:\n') ;

for k = 1:2
    if (k == 1)
        filename = sprintf ('./GB_AxB__include%d.h', k) ;
    else
        filename = sprintf ('FactoryKernels/GB_AxB__include%d.h', k) ;
    end
    fh = fopen (filename, 'w') ;
    fprintf (fh, '//------------------------------------------------------------------------------\n') ;
    fprintf (fh, '// GB_AxB__include%d.h: definitions for GB_AxB__*.c methods\n', k) ;
    fprintf (fh, '//------------------------------------------------------------------------------\n') ;
    fprintf (fh, '\n') ;
    fprintf (fh, '// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.\n') ;
    fprintf (fh, '// SPDX-License-Identifier: Apache-2.0\n\n') ;
    fprintf (fh, '// This file has been automatically generated from Generator/GB_AxB.h') ;
    fprintf (fh, '\n\n') ;
    fclose (fh) ;
end

% codegen_axb_template (multop, bmult, imult, fmult, dmult, fcmult, dcmult)

codegen_axb_template ('pair',           ...
    '1',                                ... % bool
    '1',                                ... % int, uint
    '1',                                ... % float
    '1',                                ... % double
    'GxB_CMPLXF(1,0)',                  ... % GxB_FC32_t
    'GxB_CMPLX(1,0)') ;                 ... % GxB_FC64_t

codegen_axb_template ('times',          ...
    [ ],                                ... % bool
    '(xarg*yarg)',                      ... % int, uint
    '(xarg*yarg)',                      ... % float
    '(xarg*yarg)',                      ... % double
    'GB_FC32_mul (xarg,yarg)',          ... % GxB_FC32_t
    'GB_FC64_mul (xarg,yarg)') ;        ... % GxB_FC64_t

codegen_axb_template ('first',          ...
    'xarg',                             ... % bool
    'xarg',                             ... % int, uint
    'xarg',                             ... % float
    'xarg',                             ... % double
    'xarg',                             ... % GxB_FC32_t
    'xarg') ;                           ... % GxB_FC64_t

codegen_axb_template ('second',         ...
    'yarg',                             ... % bool
    'yarg',                             ... % int, uint
    'yarg',                             ... % float
    'yarg',                             ... % double
    'yarg',                             ... % GxB_FC32_t
    'yarg') ;                           ... % GxB_FC64_t

codegen_axb_template ('min',            ...
    [ ],                                ... % bool
    'GB_IMIN (xarg,yarg)',              ... % int, uint
    'fminf (xarg,yarg)',                ... % float
    'fmin (xarg,yarg)',                 ... % double
    [ ],                                ... % GxB_FC32_t
    [ ]) ;                              ... % GxB_FC64_t

codegen_axb_template ('max',            ...
    [ ],                                ... % bool
    'GB_IMAX (xarg,yarg)',              ... % int, uint
    'fmaxf (xarg,yarg)',                ... % float
    'fmax (xarg,yarg)',                 ... % double
    [ ],                                ... % GxB_FC32_t
    [ ]) ;                              ... % GxB_FC64_t

codegen_axb_template ('plus',           ...
    [ ],                                ... % bool
    '(xarg+yarg)',                      ... % int, uint
    '(xarg+yarg)',                      ... % float
    '(xarg+yarg)',                      ... % double
    'GB_FC32_add (xarg,yarg)',          ... % GxB_FC32_t
    'GB_FC64_add (xarg,yarg)') ;        ... % GxB_FC64_t

codegen_axb_template ('minus',          ...
    [ ],                                ... % bool
    '(xarg-yarg)',                      ... % int, uint
    '(xarg-yarg)',                      ... % float
    '(xarg-yarg)',                      ... % double
    'GB_FC32_minus (xarg,yarg)',        ... % GxB_FC32_t
    'GB_FC64_minus (xarg,yarg)') ;      ... % GxB_FC64_t

codegen_axb_template ('rminus',         ...
    [ ],                                ... % bool
    '(yarg-xarg)',                      ... % int, uint
    '(yarg-xarg)',                      ... % float
    '(yarg-xarg)',                      ... % double
    'GB_FC32_minus (yarg,xarg)',        ... % GxB_FC32_t
    'GB_FC64_minus (yarg,xarg)') ;      ... % GxB_FC64_t

codegen_axb_template ('div',            ...
    [ ],                                ... % bool
    'GB_idiv (xarg,yarg)',              ... % int, uint
    '(xarg/yarg)',                      ... % float
    '(xarg/yarg)',                      ... % double
    'GB_FC32_div (xarg,yarg)',          ... % GxB_FC32_t
    'GB_FC64_div (xarg,yarg)') ;        ... % GxB_FC64_t

codegen_axb_template ('rdiv',           ...
    [ ],                                ... % bool
    'GB_idiv (yarg, xarg)',             ... % int, uint
    '(yarg/xarg)',                      ... % float
    '(yarg/xarg)',                      ... % double
    'GB_FC32_div (yarg,xarg)',          ... % GxB_FC32_t
    'GB_FC64_div (yarg,xarg)') ;        ... % GxB_FC64_t

codegen_axb_compare_template ('eq',     ...
    '(xarg == yarg)',                   ... % bool
    '(xarg == yarg)') ;                 ... % int, uint, float, double

codegen_axb_compare_template ('ne',           ...
    [ ],                                ... % bool
    '(xarg != yarg)') ;                 ... % int, uint, float, double

codegen_axb_compare_template ('gt',           ...
    '(xarg > yarg)',                    ... % bool
    '(xarg > yarg)') ;                  ... % int, uint, float, double

codegen_axb_compare_template ('lt',           ...
    '(xarg < yarg)',                    ... % bool,
    '(xarg < yarg)') ;                  ... % int, uint, float, double

codegen_axb_compare_template ('ge',           ...
    '(xarg >= yarg)',                   ... % bool
    '(xarg >= yarg)') ;                 ... % int, uint, float, double

codegen_axb_compare_template ('le',           ...
    '(xarg <= yarg)',                   ... % bool
    '(xarg <= yarg)') ;                 ... % int, uint, float, double

codegen_axb_template ('lor',            ...
    '(xarg || yarg)',                   ... % bool
    '((xarg != 0) || (yarg != 0))',     ... % int, uint
    '((xarg != 0) || (yarg != 0))',     ... % float
    '((xarg != 0) || (yarg != 0))',     ... % double
    [ ],                                ... % GxB_FC32_t
    [ ],                                ... % GxB_FC64_t
    true) ;     % no min,max,times,any monoids for LOR multiplier op

codegen_axb_template ('land',           ...
    '(xarg && yarg)',                   ... % bool
    '((xarg != 0) && (yarg != 0))',     ... % int, uint
    '((xarg != 0) && (yarg != 0))',     ... % float
    '((xarg != 0) && (yarg != 0))',     ... % double
    [ ],                                ... % GxB_FC32_t
    [ ],                                ... % GxB_FC64_t
    true) ;     % no min,max,times,any monoids for LAND multiplier op

codegen_axb_template ('lxor',           ...
    '(xarg != yarg)',                   ... % bool
    '((xarg != 0) != (yarg != 0))',     ... % int, uint
    '((xarg != 0) != (yarg != 0))',     ... % float
    '((xarg != 0) != (yarg != 0))',     ... % double
    [ ],                                ... % GxB_FC32_t
    [ ],                                ... % GxB_FC64_t
    true) ;     % no min,max,times,any monoids for LXOR multipier op

% bitwise semirings
ops   = { 'bor', 'band', 'bxor', 'bxnor' } ;
funcs = { '(xarg | yarg)', '(xarg & yarg)', '(xarg ^ yarg)', '~(xarg ^ yarg)' };
updates  = { 'warg |= targ'       , 'warg &= targ'       , 'warg ^= targ'       , 'warg = ~(warg ^ targ)'   };
ids   = {  0             , 1              , 0              , 1                };
terms = {  1             , 0              , [ ]            , [ ]              };

nbits = [8 16 32 64] ;
bits =  { '0xFF', '0xFFFF', '0xFFFFFFFF', '0xFFFFFFFFFFFFFFFFL' } ;

for i = 1:4
    addop = ops {i} ;
    fprintf ('\n%-7s', addop) ;
    addfunc = ['zarg = ' funcs{i}] ;
    update = updates {i} ;
    identity = ids {i} ;
    term = terms {i} ;
    for j = 1:4
        multop = ops {j} ;
        mult = funcs {j} ;
        for k = 1:4
            fprintf ('.') ;
            type = sprintf ('uint%d_t', nbits (k)) ;
            if (isempty (term))
                tm = [ ] ;
            elseif (term)
                tm = bits {k} ;
            else
                tm = '0' ;
            end
            if (identity)
                id = bits {k} ;
            else
                id = '0' ;
            end
            codegen_axb_method (addop, multop, update, addfunc, mult, type, type, id, tm) ;
        end
    end
end

% positional semirings
mults = { 'firsti', 'firsti1', 'firstj', 'firstj1', 'secondj', 'secondj1' } ;
multfuncs = { 'i', '(i+1)', 'k', '(k+1)', 'j', '(j+1)' } ;

% min, max, and times are normally terminal monoids, but there is no reason to terminate
% them early when used with positional operators. Only the ANY monoid is still terminal.
addops   = { 'min',                'max',                'any',   'plus',   'times'  } ;
updates  = { 'warg = GB_IMIN (warg, targ)', 'warg = GB_IMAX (warg, targ)', 'warg = targ', 'warg += targ', 'warg *= targ' } ;
addfuncs = {'GB_IMIN (xarg, yarg)','GB_IMAX (xarg, yarg)','yarg', 'xarg + yarg' , 'xarg * yarg'  } ;
ids      = { 'INT64_MAX',          'INT64_MIN',          '0',     '0',      '1'      } ;
terms    = { [ ],                  [ ],                  '0',     [ ],      [ ]      } ;

for j = 1:6
    multop = mults {j} ;
    mult = multfuncs {j} ;
    fprintf ('\n%-9s', multop) ;
    for i = 1:5
        addop = addops {i} ;
        addfunc = ['zarg = ' addfuncs{i}] ;
        update = updates {i} ;
        identity = ids {i} ;
        term = terms {i} ;
        id = ids {i} ;
        tm = terms {i} ;
        fprintf ('.') ;
        codegen_axb_method (addop, multop, update, addfunc, mult, 'int64_t', 'int64_t', id, tm) ;
        id = strrep (id, '64', '32')  ;
        fprintf ('.') ;
        codegen_axb_method (addop, multop, update, addfunc, mult, 'int32_t', 'int32_t', id, tm) ;
    end
end

fprintf ('\n') ;

