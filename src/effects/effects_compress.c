/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "effects_compress.h"
#include "../util/crc32.h"
#include "../util/rmalloc.h"

// the vendored zstd. zstd_symbols.h MUST precede zstd.h: it renames every
// global to FDB_*, so including the public header after it gives declarations
// that match the definitions in zstd.c. See src/util/zstd/PROVENANCE.md for
// why we carry our own copy rather than calling GraphBLAS's
#include "../util/zstd/zstd_symbols.h"
#include "../util/zstd/zstd.h"

#include <string.h>

//------------------------------------------------------------------------------
// little endian u32 accessors
//------------------------------------------------------------------------------
//
// Explicit rather than a memcpy of a uint32_t, because the Rust encoder writes
// these three fields with to_le_bytes and the wire is therefore defined as
// little endian rather than as native. On every target either engine builds
// for this compiles to the same load or store a memcpy would, so it costs
// nothing to be correct about it.

static inline uint32_t _read_u32le
(
	const char *p  // at least 4 readable bytes
) {
	const uint8_t *b = (const uint8_t *)p ;
	return  (uint32_t)b[0]
	     | ((uint32_t)b[1] << 8)
	     | ((uint32_t)b[2] << 16)
	     | ((uint32_t)b[3] << 24) ;
}

static inline void _write_u32le
(
	char *p,      // at least 4 writable bytes
	uint32_t v    // value to store
) {
	uint8_t *b = (uint8_t *)p ;
	b[0] = (uint8_t)( v        & 0xFF) ;
	b[1] = (uint8_t)((v >> 8)  & 0xFF) ;
	b[2] = (uint8_t)((v >> 16) & 0xFF) ;
	b[3] = (uint8_t)((v >> 24) & 0xFF) ;
}

//------------------------------------------------------------------------------
// write
//------------------------------------------------------------------------------

bool EffectsV3_MaybeCompress
(
	char **payload,   // [input/output] complete payload, replaced on success
	size_t *len,      // [input/output] payload length
	size_t min_bytes  // smallest record stream worth compressing, 0 disables
) {
	ASSERT (payload != NULL) ;
	ASSERT (*payload != NULL) ;
	ASSERT (len != NULL) ;

	char   *buff = *payload ;
	size_t  n    = *len ;

	// compression disabled, or nothing to compress
	if (min_bytes == 0 || n < EFFECTS_V3_HEADER_LEN) {
		return false ;
	}

	// the threshold is measured against the RECORD STREAM, not the whole
	// payload - the two header bytes are not what an operator is trading
	// bandwidth for
	size_t records_len = n - EFFECTS_V3_HEADER_LEN ;
	if (records_len < min_bytes) {
		return false ;
	}

	// already compressed. See the header for why this is a refusal rather
	// than an ASSERT
	if ((uint8_t)buff[1] & EFFECTS_V3_FLAG_COMPRESSED) {
		return false ;
	}

	const char *records = buff + EFFECTS_V3_HEADER_LEN ;

	size_t  bound = ZSTD_compressBound (records_len) ;
	char   *frame = rm_malloc (bound) ;

	size_t frame_len = ZSTD_compress (frame, bound, records, records_len,
			EFFECTS_V3_COMPRESSION_LEVEL) ;

	// a compression failure is not a reason to fail the write: the
	// uncompressed payload is still correct
	if (ZSTD_isError (frame_len)) {
		rm_free (frame) ;
		return false ;
	}

	// the two declared lengths and the checksum ride along, so they count
	if (!EffectsV3_CompressionWorthIt (frame_len, records_len)) {
		rm_free (frame) ;
		return false ;
	}

	size_t  out_len = EFFECTS_V3_HEADER_LEN + EFFECTS_V3_COMPRESSED_PREFIX
	                + frame_len ;
	char   *out     = rm_malloc (out_len) ;

	out[0] = buff[0] ;                                        // version
	out[1] = (char)((uint8_t)buff[1] | EFFECTS_V3_FLAG_COMPRESSED) ;

	_write_u32le (out + 2,  (uint32_t)records_len) ;           // plain length
	_write_u32le (out + 6,  (uint32_t)frame_len) ;             // frame length
	_write_u32le (out + 10, CRC32 (records, records_len)) ;    // of the PLAINTEXT
	memcpy (out + EFFECTS_V3_HEADER_LEN + EFFECTS_V3_COMPRESSED_PREFIX,
			frame, frame_len) ;

	rm_free (frame) ;
	rm_free (buff) ;

	*payload = out ;
	*len     = out_len ;

	return true ;
}

//------------------------------------------------------------------------------
// read
//------------------------------------------------------------------------------

const char *EffectsV3CompressFault_ToString
(
	EffectsV3CompressFault fault
) {
	switch (fault) {
		case EFFECTS_V3_COMPRESS_OK:
			return "ok" ;
		case EFFECTS_V3_COMPRESS_SHORT_PREFIX:
			return "payload ends inside the compressed prefix" ;
		case EFFECTS_V3_COMPRESS_SHORT_FRAME:
			return "declared frame length exceeds the bytes available" ;
		case EFFECTS_V3_COMPRESS_TRAILING:
			return "bytes follow the compressed frame" ;
		case EFFECTS_V3_COMPRESS_BAD_FRAME:
			return "zstd refused the frame" ;
		case EFFECTS_V3_COMPRESS_LENGTH_MISMATCH:
			return "frame expands to a different length than declared" ;
		case EFFECTS_V3_COMPRESS_CHECKSUM:
			return "plaintext checksum disagrees with the header" ;
		default:
			return "unknown" ;
	}
}

EffectsV3Status EffectsV3_OpenCompressed
(
	const char *body,                 // compressed prefix followed by the frame
	size_t body_len,                  // bytes available from 'body'
	char **plain,                     // [output] inflated record stream
	size_t *plain_len,                // [output] its length
	EffectsV3CompressFault *fault     // [output, optional] which check fired
) {
	ASSERT (body != NULL) ;
	ASSERT (plain != NULL) ;
	ASSERT (plain_len != NULL) ;

	*plain     = NULL ;
	*plain_len = 0 ;

	EffectsV3CompressFault _fault = EFFECTS_V3_COMPRESS_OK ;
	#define FAIL(f, status)                      \
		do {                                     \
			if (fault != NULL) *fault = (f) ;    \
			return (status) ;                    \
		} while (0)

	if (fault != NULL) *fault = _fault ;

	// the prefix itself has to be there before any of it can be read
	if (body_len < EFFECTS_V3_COMPRESSED_PREFIX) {
		FAIL (EFFECTS_V3_COMPRESS_SHORT_PREFIX, EFFECTS_V3_TRUNCATED) ;
	}

	uint32_t declared_plain = _read_u32le (body) ;
	uint32_t declared_comp  = _read_u32le (body + 4) ;
	uint32_t declared_crc   = _read_u32le (body + 8) ;

	const char *frame = body + EFFECTS_V3_COMPRESSED_PREFIX ;
	size_t      avail = body_len - EFFECTS_V3_COMPRESSED_PREFIX ;

	// exactly 'declared_comp', not "the rest": the header states the frame's
	// own length, so the payload is self-delimiting and the decompress input
	// is bounded from the header rather than from wherever the buffer happens
	// to end
	if (avail < declared_comp) {
		FAIL (EFFECTS_V3_COMPRESS_SHORT_FRAME, EFFECTS_V3_TRUNCATED) ;
	}

	// trailing bytes mean the header and the payload disagree about where the
	// frame ends, and nothing after it can be trusted either
	if (avail > declared_comp) {
		FAIL (EFFECTS_V3_COMPRESS_TRAILING, EFFECTS_V3_MALFORMED) ;
	}

	// 'declared_plain' is the allocation CEILING and it is applied by passing
	// it as zstd's destination capacity: a frame that wants to expand beyond
	// it is refused by zstd before writing anything, rather than after an
	// allocation that has already happened. That distinction is the whole
	// point - zstd's ratio on repetitive input is unbounded in practice, so a
	// decoder that grows to fit whatever the frame expands to can be made to
	// allocate gigabytes from a hundred byte payload.
	//
	// rm_malloc of 1 rather than 0 for an empty plaintext, so the pointer is
	// distinguishable from the failure case. A zero length plaintext cannot
	// be produced by EffectsV3_MaybeCompress - it would not clear the
	// worth-it test - but the Rust reader accepts one, and refusing what the
	// peer accepts is divergence
	char *out = rm_malloc (declared_plain > 0 ? declared_plain : 1) ;

	size_t got = ZSTD_decompress (out, declared_plain, frame, declared_comp) ;

	if (ZSTD_isError (got)) {
		rm_free (out) ;
		FAIL (EFFECTS_V3_COMPRESS_BAD_FRAME, EFFECTS_V3_MALFORMED) ;
	}

	// still cross-checked after the fact, because the ceiling is an UPPER
	// bound: a frame that expands to LESS than declared satisfies the
	// capacity and still means the header and the payload disagree. The two
	// checks fail on different inputs, which is why both are here
	if (got != (size_t)declared_plain) {
		rm_free (out) ;
		FAIL (EFFECTS_V3_COMPRESS_LENGTH_MISMATCH, EFFECTS_V3_MALFORMED) ;
	}

	// over the PLAINTEXT, not the frame: these are the bytes records are
	// parsed from, and it stays reproducible across zstd versions, which may
	// frame the same input differently
	if (CRC32 (out, got) != declared_crc) {
		rm_free (out) ;
		FAIL (EFFECTS_V3_COMPRESS_CHECKSUM, EFFECTS_V3_MALFORMED) ;
	}

	#undef FAIL

	*plain     = out ;
	*plain_len = got ;

	return EFFECTS_V3_OK ;
}
