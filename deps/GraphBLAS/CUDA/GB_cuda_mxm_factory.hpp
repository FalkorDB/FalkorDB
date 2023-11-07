//------------------------------------------------------------------------------
// GB_cuda_mxm_factory: construct code and header file for GrB_mxm jit kernel
//------------------------------------------------------------------------------

// Class to manage both stringify functions from mxm, ops and monoids to a
// header file.

// (c) Nvidia Corp. 2020 All rights reserved
// SPDX-License-Identifier: Apache-2.0

// Implementations of string callbacks
#pragma once

// FIXME do we need the iostrean any more?
#include <iostream>
#include <cstdint>
#include "GB_jit_cache.h"

extern "C"
{
    #include "GB.h"
    #include "GB_binop.h"
    #include "GB_stringify.h"
}

// FIXME: do we need the file_callback method?
// Define function pointer we will use later
//std::istream* (*file_callback)(std::string, std::iostream&);

//------------------------------------------------------------------------------
// GB_cuda_mxm_factory
//------------------------------------------------------------------------------

// Define a factory class for building any mxm text definitions

class GB_cuda_mxm_factory: public jit::File_Desc
{

    //--------------------------------------------------------------------------
    // public members of the object
    //--------------------------------------------------------------------------

    public:

        uint64_t sr_code ;          // unique 62-bit code for a GrB_mxm problem
        GrB_Semiring semiring ;     // the semiring for GrB_mxm
        GrB_Type ctype, atype, btype ;  // the types of C, A, and B
        FILE *fp ;                  // file for GB_mxm_*.h header

    //--------------------------------------------------------------------------
    // open/close: access the GB_mxm_*.h header file for a specific instance
    //--------------------------------------------------------------------------

    void open (const char *path_and_file, const char *mode)
    {
        fp = fopen (path_and_file, mode) ;
    }

    void close( )
    {
        fclose (fp) ;
    }

    //--------------------------------------------------------------------------
    // mxm_factory: create unique code for a GrB_mxm problem
    //--------------------------------------------------------------------------

    // mxm_factory takes a set of inputs describing and operation (semiring,
    // mask, datatypes, sparsity formats, etc) and produces a numerical unique
    // value for those. This allows rapid lookups to see if we have handled this
    // case before, and avoids the need to generate and manage strings at this
    // stage.

    // FIXME: pass in user's C_in matrix, in case C_in<M>+=A*B can be done
    //        in-place
    // FIXME: handle hypersparse case in dot3

    void mxm_factory
    (
        // C matrix:
        bool C_iso,             // true if C is iso-valued
        bool C_in_iso,          // C input iso status
        int C_sparsity,         // sparsity structure of C
        GrB_Type ctype,         // the type of C
        // M matrix:
        GrB_Matrix M,           // may be NULL
        bool Mask_struct,       // mask is structural
        bool Mask_comp,         // mask is complemented
        // semiring:
        GrB_Semiring semiring,  // the semiring to enumify
        bool flipxy,            // multiplier is: mult(a,b) or mult(b,a)
        // A and B:
        GrB_Matrix A,
        GrB_Matrix B
    )
    {

        if (C_iso)
        {
            // the kernel does not access any values of C, A, or B
            semiring = GxB_ANY_PAIR_BOOL ;
            flipxy = false ;
        }

       uint64_t scode ;

       GB_enumify_mxm (
	    // output:
	    &scode,         // unique encoding of the entire semiring
	    // input:
            C_iso,          // true if C is iso-valued
            C_in_iso,
	    C_sparsity,     // sparsity structure of C
	    ctype,          // the type of C
            // M matrix:
            M,
	    Mask_struct,    // mask is structural
	    Mask_comp,      // mask is complemented
            // semiring:
	    semiring,      // the semiring to enumify
	    flipxy,        // multiplier is: mult(a,b) or mult(b,a)
            // A and B:
            A,
            B
       ) ;

       this->sr_code = scode;
       this->semiring = semiring ;
       this->atype = A->type ;
       this->btype = B->type ;
       this->ctype = ctype ;

       std::stringstream ss;
       // FIXME: use same name scheme as the CPU jit
       ss << "GB_mxm_" << this->sr_code << ".h";

       std::string new_filename = ss.str();
       filename.resize(new_filename.size());
       strcpy(filename.data(), new_filename.data());

    }

    //--------------------------------------------------------------------------
    // macrofy: create macros from sr_code and data types
    //--------------------------------------------------------------------------

    // macrofy takes a code and creates the corresponding string macros for
    // operators, datatypes, sparsity formats and writes its results to a file.

    void macrofy ( ) override
    {
       GB_macrofy_mxm (
	    // output to file :
	    fp,
	    // input:
	    this->sr_code,
	    this->semiring,
	    this->ctype,
	    this->atype,
	    this->btype
       ) ;
    }

} ; // GB_cuda_mxm_factory

