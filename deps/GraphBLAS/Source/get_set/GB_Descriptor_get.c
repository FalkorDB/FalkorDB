//------------------------------------------------------------------------------
// GB_Descriptor_get: get the status of a descriptor
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// A descriptor modifies the behavoir of a GraphBLAS operation.

// This function is called via the GB_GET_DESCRIPTOR(...) macro.

//  Descriptor field:           Descriptor value:

//  desc->out                   GxB_DEFAULT or GrB_REPLACE

//      GrB_REPLACE means that the output matrix C is cleared just
//      prior to writing results back into it, via C<M> = results.  This
//      descriptor does not affect how C is used to compute the results.  If
//      GxB_DEFAULT, then C is not cleared before doing C<M>=results.

//  desc->mask                  GxB_DEFAULT, GrB_COMP, GrB_STRUCTURE, or
//                              GrB_COMP + GrB_STRUCTURE

//      An optional 'write mask' defines how the results are to be written back
//      into C.  The boolean mask matrix M has the same size as C (M is
//      typecasted to boolean if it has another type).  If the M input to the
//      GraphBLAS method is NULL, then implicitly M(i,j)=1 for all i and j.
//      Let Z be the results to be written into C (the same dimension as C).
//      If desc->mask is GxB_DEFAULT, and M(i,j)=1, then C(i,j) is over-written
//      with Z(i,j).  Otherwise, if M(i,j)=0 C(i,j) is left unmodified (it
//      remains an implicit zero if it is so, or its value is unchanged if it
//      has one).  If desc->mask is GrB_COMP, then the use of M is negated:
//      M(i,j)=0 means that C(i,j) is overwritten with Z(i,j), and M(i,j)=1
//      means that C(i,j) is left unchanged.  If the value is GrB_STRUCTURE,
//      only the pattern is used; any entry present in the pattern has the
//      value M(i,j)=1, and entries not in the pattern have the value M(i,j)=0.
//      The GrB_COMP and GrB_STUCTURE options can be combined, as GrB_COMP +
//      GrB_STRUCTURE.

//      Writing results Z into C via the mask M is written as C<M>=Z in
//      GraphBLAS notation.

//      Note that it is the value of M(i,j) that determines how C(i,j) is
//      overwritten.  If the (i,j) entry is present in the M matrix data
//      structure but has a numerical value of zero, then it is the same as if
//      (i,j) is not present and thus implicitly zero.  Both mean 'M(i,j)=0'
//      in the description above of how the mask M works.

//  desc->in0 and desc->in1     GxB_DEFAULT or GrB_TRAN

//      A GrB_Matrix passed as an input parameter to GraphBLAS methods can
//      optionally transpose them prior to using them.  desc->in0 always refers
//      to the first input to the method, and desc->in1 always refers to the
//      second one.

//      If the value of this descriptor is GxB_DEFAULT, then the matrix is used
//      as-is.  Otherwise, it is transposed first.  That is, the results are
//      the same as if the transpose of the matrix was passed to the method.

//  desc->axb                   can be:

//      GxB_DEFAULT = 0         automatic selection
//      GxB_AxB_GUSTAVSON       gather-scatter saxpy method
//      GxB_AxB_HASH            hash-based saxpy method
//      GxB_AxB_SAXPY           saxpy: either Gustavson or hash
//      GxB_AxB_DOT             dot product

//  desc->do_sort               true or false (default is false) 

#include "GB.h"

GrB_Info GB_Descriptor_get      // get the contents of a descriptor
(
    const GrB_Descriptor desc,  // descriptor to query, may be NULL
    bool *C_replace,            // if true replace C before C<M>=Z
    bool *Mask_comp,            // if true use logical negation of M
    bool *Mask_struct,          // if true use the structure of M
    bool *In0_transpose,        // if true transpose first input
    bool *In1_transpose,        // if true transpose second input
    GrB_Desc_Value *AxB_method, // method for C=A*B
    int *do_sort                // if nonzero, sort in GrB_mxm
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    // desc may be null, but if not NULL it must be initialized
    GB_RETURN_IF_FAULTY (desc) ;

    //--------------------------------------------------------------------------
    // get the contents of the descriptor
    //--------------------------------------------------------------------------

    // default values if descriptor is NULL
    GrB_Desc_Value C_desc    = GxB_DEFAULT ;
    GrB_Desc_Value Mask_desc = GxB_DEFAULT ;
    GrB_Desc_Value In0_desc  = GxB_DEFAULT ;
    GrB_Desc_Value In1_desc  = GxB_DEFAULT ;
    GrB_Desc_Value AxB_desc  = GxB_DEFAULT ;

    int do_sort_desc         = GxB_DEFAULT ;

    // non-defaults descriptor values
    if (desc != NULL)
    { 
        // get the contents
        C_desc    = desc->out ;   // DEFAULT or REPLACE
        Mask_desc = desc->mask ;  // DEFAULT, COMP, STRUCTURE, or COMP+STRUCTURE
        In0_desc  = desc->in0 ;   // DEFAULT or TRAN
        In1_desc  = desc->in1 ;   // DEFAULT or TRAN
        AxB_desc  = desc->axb ;   // DEFAULT, GUSTAVSON, HASH, or DOT
        do_sort_desc = desc->do_sort ;  // DEFAULT, or true (nonzero)
    }

    // check for valid values of each descriptor field
    if (!(C_desc    == GxB_DEFAULT || C_desc    == GrB_REPLACE) ||
        !(Mask_desc == GxB_DEFAULT   || Mask_desc == GrB_COMP ||
          Mask_desc == GrB_STRUCTURE || Mask_desc == GrB_COMP+GrB_STRUCTURE) ||
        !(In0_desc  == GxB_DEFAULT || In0_desc  == GrB_TRAN) ||
        !(In1_desc  == GxB_DEFAULT || In1_desc  == GrB_TRAN) ||
        !(AxB_desc  == GxB_DEFAULT || AxB_desc  == GxB_AxB_GUSTAVSON ||
          AxB_desc  == GxB_AxB_DOT ||
          AxB_desc  == GxB_AxB_HASH || AxB_desc  == GxB_AxB_SAXPY))
    { 
        return (GrB_INVALID_OBJECT) ;
    }

    if (C_replace != NULL)
    { 
        *C_replace = (C_desc == GrB_REPLACE) ;
    }
    if (Mask_comp != NULL)
    { 
        *Mask_comp = (Mask_desc == GrB_COMP)
                  || (Mask_desc == GrB_COMP + GrB_STRUCTURE) ;
    }
    if (Mask_struct != NULL)
    { 
        *Mask_struct = (Mask_desc == GrB_STRUCTURE)
                    || (Mask_desc == GrB_STRUCTURE + GrB_COMP) ;
    }
    if (In0_transpose != NULL)
    { 
        *In0_transpose = (In0_desc == GrB_TRAN) ;
    }
    if (In1_transpose != NULL)
    { 
        *In1_transpose = (In1_desc == GrB_TRAN) ;
    }
    if (AxB_method != NULL)
    { 
        *AxB_method = AxB_desc ;
    }
    if (do_sort != NULL)
    { 
        *do_sort = do_sort_desc ;
    }

    return (GrB_SUCCESS) ;
}

