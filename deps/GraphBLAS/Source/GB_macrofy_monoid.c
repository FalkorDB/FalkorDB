//------------------------------------------------------------------------------
// GB_macrofy_monoid: build macros for a monoid
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_stringify.h"

void GB_macrofy_monoid  // construct the macros for a monoid
(
    FILE *fp,           // File to write macros, assumed open already
    // inputs:
    int add_ecode,      // binary op as an enum
    int id_ecode,       // identity value as an enum
    int term_ecode,     // terminal value as an enum (<= 28 is terminal)
    bool C_iso,         // true if C is iso
    GrB_Monoid monoid,  // monoid to macrofy
    bool disable_terminal_condition,    // if true, a builtin monoid is assumed
                        // to be non-terminal.  For the (times, firstj, int64)
                        // semiring, times is normally a terminal monoid, but
                        // it's not worth exploiting in GrB_mxm.
    // output:
    const char **u_expression
)
{

    GrB_BinaryOp op = monoid->op ;
    const char *ztype_name = C_iso ? "void" : op->ztype->name ;
    int zcode = C_iso ? 0 : op->ztype->code ;
    size_t zsize = C_iso ? 0 : op->ztype->size ;
    GB_Opcode opcode = C_iso ? 0 : op->opcode ;

    //--------------------------------------------------------------------------
    // create macros for the additive operator
    //--------------------------------------------------------------------------

    GB_macrofy_binop (fp, "GB_ADD", false, true, false, add_ecode, C_iso,
        op, NULL, u_expression) ;

    //--------------------------------------------------------------------------
    // create macros for the identity value
    //--------------------------------------------------------------------------

    bool has_byte ;
    uint8_t byte ;
    if (C_iso)
    { 
        // no values computed (C is iso)
        fprintf (fp, "#define GB_DECLARE_IDENTITY(z)\n") ;
        fprintf (fp, "#define GB_DECLARE_IDENTITY_CONST(z)\n") ;
    }
    else if (id_ecode <= 28)
    {
        // built-in monoid: a simple assignment
        const char *id_val = GB_macrofy_id (id_ecode, zsize, &has_byte, &byte) ;
        #define SLEN (256 + GxB_MAX_NAME_LEN)
        char id [SLEN] ;
        if (zcode == GB_FC32_code)
        { 
            snprintf (id, SLEN, "%s z = GxB_CMPLXF (%s,0)",
                ztype_name, id_val) ;
        }
        else if (zcode == GB_FC64_code)
        { 
            snprintf (id, SLEN, "%s z = GxB_CMPLX (%s,0)",
                ztype_name, id_val) ;
        }
        else
        { 
            snprintf (id, SLEN, "%s z = %s", ztype_name, id_val) ;
        }
        fprintf (fp, "#define GB_DECLARE_IDENTITY(z) %s\n", id) ;
        fprintf (fp, "#define GB_DECLARE_IDENTITY_CONST(z) const %s\n", id) ;
        if (has_byte)
        { 
            fprintf (fp, "#define GB_HAS_IDENTITY_BYTE 1\n") ;
            fprintf (fp, "#define GB_IDENTITY_BYTE 0x%02x\n", (int) byte) ;
        }
    }
    else
    { 
        // user-defined monoid:  all we have are the bytes
        GB_macrofy_bytes (fp, "IDENTITY", "z",
            ztype_name, (uint8_t *) (monoid->identity), zsize, true) ;
        fprintf (fp, "#define GB_DECLARE_IDENTITY_CONST(z) "
            "GB_DECLARE_IDENTITY(z)\n") ;
    }

    //--------------------------------------------------------------------------
    // create macros for the terminal value and terminal conditions
    //--------------------------------------------------------------------------

    // GB_TERMINAL_CONDITION(z,zterminal) should return true if the value of z
    // has reached its terminal value (zterminal), or false otherwise.  If the
    // monoid is not terminal, then the macro should always return false.  The
    // ANY monoid should always return true.

    // GB_IF_TERMINAL_BREAK(z,zterminal) is a macro containing a full
    // statement.  If the monoid is never terminal, it becomes the empty
    // statement.  Otherwise, it checks the terminal condition and does a
    // "break" if true.

    // GB_DECLARE_TERMINAL_CONST(zterminal) declares the zterminal variable as
    // const.  It is empty if the monoid is not terminal.

    bool monoid_is_terminal = false ;

    bool is_any_monoid = (term_ecode == 18) ;
    if (is_any_monoid || C_iso)
    { 
        // ANY monoid is terminal but with no specific terminal value
        fprintf (fp, "#define GB_IS_ANY_MONOID 1\n") ;
        monoid_is_terminal = true ;
    }
    else if (monoid->terminal == NULL)
    { 
        // monoid is not terminal (either built-in or user-defined)
        monoid_is_terminal = false ;
    }
    else if (term_ecode <= 28)
    {
        // built-in terminal monoid: terminal value is a simple assignment.
        // Its terminal condition can be ignored (for (times, firstj, int64),
        // for example) if disable_terminal_condition is true, but in that case
        // its terminal value must still be constructed for the query function.
        monoid_is_terminal = !disable_terminal_condition ;
        if (monoid_is_terminal)
        { 
            fprintf (fp, "#define GB_MONOID_IS_TERMINAL 1\n") ;
        }
        // there are no built-in terminal monoids, except ANY, handled above
        ASSERT (zcode != GB_FC32_code && zcode != GB_FC64_code) ;
        const char *term_value = GB_macrofy_id (term_ecode, zsize, NULL, NULL) ;
        fprintf (fp, "#define GB_DECLARE_TERMINAL_CONST(zterminal) "
            "const %s zterminal = ", ztype_name) ;
        fprintf (fp, "%s\n", term_value) ;
        if (monoid_is_terminal)
        { 
            fprintf (fp, "#define GB_TERMINAL_CONDITION(z,zterminal) "
                "((z) == %s)\n", term_value) ;
            fprintf (fp, "#define GB_IF_TERMINAL_BREAK(z,zterminal) "
                "if ((z) == %s) break\n", term_value) ;
        }
    }
    else
    { 
        // user-defined terminal monoid
        monoid_is_terminal = true ;
        fprintf (fp, "#define GB_MONOID_IS_TERMINAL 1\n") ;
        GB_macrofy_bytes (fp, "TERMINAL_CONST", "zterminal",
            ztype_name, monoid->terminal, zsize, false) ;
        fprintf (fp, "#define GB_TERMINAL_CONDITION(z,zterminal) "
            " (memcmp (&(z), &(zterminal), %d) == 0)\n", (int) zsize) ;
        fprintf (fp, "#define GB_IF_TERMINAL_BREAK(z,zterminal) "
            " if (memcmp (&(z), &(zterminal), %d) == 0) break\n", (int) zsize) ;
    }

    //--------------------------------------------------------------------------
    // determine the OpenMP #pragma omp reduction(redop:z) for this monoid
    //--------------------------------------------------------------------------

    // If not #define'd, the default in GB_monoid_shared_definitions.h is no
    // #pragma.  The pragma is empty if the monoid is terminal, since the simd
    // reduction does not work with a 'break' in the loop.

    bool is_complex = (zcode == GB_FC32_code || zcode == GB_FC64_code) ;

    if (is_complex)
    { 
        fprintf (fp, "#define GB_Z_IS_COMPLEX 1\n") ;
    }

    if (!monoid_is_terminal && !is_complex)
    {
        char *redop = NULL ;
        if (opcode == GB_PLUS_binop_code)
        { 
            // #pragma omp simd reduction(+:z)
            redop = "+" ;
        }
        else if (opcode == GB_LXOR_binop_code || opcode == GB_BXOR_binop_code)
        { 
            // #pragma omp simd reduction(^:z)
            redop = "^" ;
        }
        else if (opcode == GB_TIMES_binop_code)
        { 
            // #pragma omp simd reduction(^:z)
            redop = "*" ;
        }
        if (redop != NULL)
        { 
            // The monoid has a "#pragma omp simd reduction(redop:z)" statement.
            // There are other OpenMP reductions that could be exploited, but
            // many are for terminal monoids (logical and bitwise AND, OR).
            // The min/max reductions are not exploited because they are
            // terminal monoids for integers.  For floating-point, the NaN
            // handling may differ, so they are not exploited here either.
            fprintf (fp, "#define GB_PRAGMA_SIMD_REDUCTION_MONOID(z) "
                "GB_PRAGMA_SIMD_REDUCTION (%s,z)\n", redop) ;
        }
    }

    //--------------------------------------------------------------------------
    // special cases
    //--------------------------------------------------------------------------

    bool is_integer = (zcode >= GB_INT8_code && zcode <= GB_UINT64_code) ;
    bool is_fp_real = (zcode == GB_FP32_code || zcode == GB_FP64_code) ;

    if (opcode == GB_PLUS_binop_code && zcode == GB_FC32_code)
    { 
        // PLUS_FC32 monoid
        fprintf (fp, "#define GB_IS_PLUS_FC32_MONOID 1\n") ;
    }
    else if (opcode == GB_PLUS_binop_code && zcode == GB_FC64_code)
    { 
        // PLUS_FC64 monoid
        fprintf (fp, "#define GB_IS_PLUS_FC64_MONOID 1\n") ;
    }
    else if (opcode == GB_MIN_binop_code && is_integer)
    { 
        // IMIN monoid (min with any integer type)
        fprintf (fp, "#define GB_IS_IMIN_MONOID 1\n") ;
    }
    else if (opcode == GB_MAX_binop_code && is_integer)
    { 
        // IMAX monoid (max with any integer type)
        fprintf (fp, "#define GB_IS_IMAX_MONOID 1\n") ;
    }
    else if (opcode == GB_MIN_binop_code && is_fp_real)
    { 
        // FMIN monoid (min with a real floating-point type)
        fprintf (fp, "#define GB_IS_FMIN_MONOID 1\n") ;
    }
    else if (opcode == GB_MAX_binop_code && is_fp_real)
    { 
        // FMAX monoid (max with a real floating-point type)
        fprintf (fp, "#define GB_IS_FMAX_MONOID 1\n") ;
    }

    // can ignore overflow in ztype when accumulating the result via the monoid
    // zcode == 0: only when C is iso
    bool ztype_ignore_overflow = (zcode == 0 ||
        zcode == GB_INT64_code || zcode == GB_UINT64_code ||
        zcode == GB_FP32_code  || zcode == GB_FP64_code ||
        zcode == GB_FC32_code  || zcode == GB_FC64_code) ;
    if (ztype_ignore_overflow && !is_any_monoid)
    { 
        // if the monoid is ANY, this is set to 1 by
        // GB_monoid_shared_definitions.h, so skip it here
        fprintf (fp, "#define GB_Z_IGNORE_OVERFLOW 1\n") ;
    }

    //--------------------------------------------------------------------------
    // create macros for atomics on the CPU
    //--------------------------------------------------------------------------

    fprintf (fp, "#define GB_Z_NBITS %d\n", 8 * (int) zsize) ;

    // atomic write
    bool has_atomic_write = false ;
    char *ztype_atomic = NULL ;
    if (zcode == 0)
    { 
        // C is iso (any_pair symbolic semiring)
        fprintf (fp, "#define GB_Z_ATOMIC_BITS 0\n") ;
    }
    else if (zsize == sizeof (uint8_t))
    { 
        // int8_t, uint8_t, and 8-bit user-defined types
        ztype_atomic = "uint8_t" ;
        has_atomic_write = true ;
        fprintf (fp, "#define GB_Z_ATOMIC_BITS 8\n") ;
    }
    else if (zsize == sizeof (uint16_t))
    { 
        // int16_t, uint16_t, and 16-bit user-defined types
        ztype_atomic = "uint16_t" ;
        has_atomic_write = true ;
        fprintf (fp, "#define GB_Z_ATOMIC_BITS 16\n") ;
    }
    else if (zsize == sizeof (uint32_t))
    { 
        // int32_t, uint32_t, float, and 32-bit user-defined types
        ztype_atomic = "uint32_t" ;
        has_atomic_write = true ;
        fprintf (fp, "#define GB_Z_ATOMIC_BITS 32\n") ;
    }
    else if (zsize == sizeof (uint64_t))
    { 
        // int64_t, uint64_t, double, float complex, and 64-bit user types
        ztype_atomic = "uint64_t" ;
        has_atomic_write = true ;
        fprintf (fp, "#define GB_Z_ATOMIC_BITS 64\n") ;
    }

    // atomic write for the ztype:  if GB_Z_ATOMIC_BITS is defined, then
    // GB_Z_HAS_ATOMIC_WRITE is #defined as 1 by GB_kernel_shared_definitions.h
    if (has_atomic_write && (zcode == GB_FC32_code || zcode == GB_UDT_code))
    { 
        // user-defined types of size 1, 2, 4, or 8 bytes can be written
        // atomically, but must use a pun with ztype_atomic.  float complex
        // should also ztype_atomic.
        fprintf (fp, "#define GB_Z_ATOMIC_TYPE %s\n", ztype_atomic) ;
    }

    // OpenMP atomic update support
    bool is_real = (zcode >= GB_BOOL_code && zcode <= GB_FP64_code) ;
    bool has_atomic_update = false ;
    int omp_atomic_version = 0 ;

    switch (opcode)
    {

        case GB_ANY_binop_code   : 
            // the ANY monoid is a special case.  It is done with an atomic
            // write, or no update at all.  The atomic write can be done for
            // float complex (64 bits) but not double complex (128 bits).
            // The atomic update is identical: just an atomic write.
            has_atomic_update = has_atomic_write ;
            omp_atomic_version = is_real ? 2 : 0 ;
            break ;

        case GB_LAND_binop_code  : 
        case GB_LOR_binop_code   : 
        case GB_LXOR_binop_code  : 
        case GB_BAND_binop_code  : 
        case GB_BOR_binop_code   : 
        case GB_BXOR_binop_code  : 
            // OpenMP 4.0 atomic, not on MS Visual Studio
            has_atomic_update = true ;
            omp_atomic_version = 4 ;
            break ;

        case GB_BXNOR_binop_code : 
        case GB_EQ_binop_code    : // LXNOR
        case GB_MIN_binop_code   : 
        case GB_MAX_binop_code   : 
            // these monoids can be done via atomic compare/exchange,
            // but not with an omp pragma
            has_atomic_update = true ;
            break ;

        case GB_PLUS_binop_code  : 
            // even complex can be done atomically
            has_atomic_update = true ;
            omp_atomic_version = 2 ;
            break ;

        case GB_TIMES_binop_code : 
            // real monoids can be done atomically, not double complex
            has_atomic_update = is_real || (zcode == GB_FC32_code) ;
            // only the real case has an omp pragma
            omp_atomic_version = is_real ? 2 : 0 ;
            break ;

        default : 
            // all other monoids, including user-defined, can be done atomically
            // via compare-and-swap, if z has size 1, 2, 4, or 8 bytes.
            // Otherwise, they must be done in a critical section.
            has_atomic_update = has_atomic_write ;
            omp_atomic_version = 0 ;
    }

    if (has_atomic_update)
    { 
        // the monoid can be done atomically
        fprintf (fp, "#define GB_Z_HAS_ATOMIC_UPDATE 1\n") ;
        if (omp_atomic_version == 4)
        { 
            // OpenMP 4.0 has an omp pragram but not OpenMP 2.0. 
            fprintf (fp, "#define GB_Z_HAS_OMP_ATOMIC_UPDATE "
                "(!GB_COMPILER_MSC)\n") ;
        }
        else if (omp_atomic_version == 2)
        { 
            // this update has an omp pragm 
            fprintf (fp, "#define GB_Z_HAS_OMP_ATOMIC_UPDATE 1\n") ;
        }
    }

    //--------------------------------------------------------------------------
    // create macros for the atomic CUDA operator, if available
    //--------------------------------------------------------------------------

    const char *a = NULL, *cuda_type = NULL ;
    bool user_monoid_atomically = false ;
    GB_enumify_cuda_atomic (&a, &user_monoid_atomically, &cuda_type,
        monoid, add_ecode, zsize, zcode) ;

    if (monoid == NULL || zcode == 0)
    { 
        // nothing to do: C is iso-valued.  For GrB_mxm only.
        ;
    }
    else if (user_monoid_atomically)
    { 
        // CUDA atomic for a user monoid
        fprintf (fp, "#define GB_Z_HAS_CUDA_ATOMIC_USER 1\n") ;
        fprintf (fp, "#define GB_Z_CUDA_ATOMIC_TYPE %s\n", cuda_type) ;
    }
    else if (a == NULL)
    { 
        // no CUDA atomics for this monoid
        ;
    }
    else
    { 
        // CUDA atomic available for a built-in monoid
        fprintf (fp, "#define GB_Z_HAS_CUDA_ATOMIC_BUILTIN 1\n") ;
        fprintf (fp, "#define GB_Z_CUDA_ATOMIC %s\n", a) ;
        fprintf (fp, "#define GB_Z_CUDA_ATOMIC_TYPE %s\n", cuda_type) ;
    }
}

