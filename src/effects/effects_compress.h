/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "effects_v3.h"

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

//------------------------------------------------------------------------------
// effects v3 - payload compression
//------------------------------------------------------------------------------
//
// The v3 header is two bytes, and a COMPRESSED payload adds twelve more:
//
//     u8  version = 3
//     u8  flags                    bit 0 = compressed
//     u32 uncompressed_length      only when compressed
//     u32 compressed_length        only when compressed
//     u32 checksum                 only when compressed, CRC-32 of the PLAINTEXT
//     records... | zstd frame
//
// The header is never compressed, so a reader knows what it holds before
// decoding anything. Three rules follow:
//
//   * compression must save MORE than the twelve byte prefix to be worth doing
//   * `compressed_length` makes the payload self-delimiting, so TRAILING BYTES
//     ARE A DECODE ERROR rather than tolerated padding
//   * `uncompressed_length` is the allocation CEILING, applied before the
//     allocation and cross-checked after - both, because they fail on
//     different inputs (see EffectsV3_OpenCompressed)
//
// The checksum covers the PLAINTEXT, not the frame: those are the bytes
// records parse from, and it stays reproducible across zstd versions.
//
// `EFFECTS_COMPRESSION` is a byte threshold, default 0 = off.
//
//------------------------------------------------------------------------------

// flags bit 0: the records are a zstd frame rather than a record stream
#define EFFECTS_V3_FLAG_COMPRESSED 0x01

// every flag bit this build understands. A reader meeting anything outside
// this mask REFUSES the buffer rather than masking it off: decoding anyway
// would apply a prefix of something whose shape it does not know
#define EFFECTS_V3_KNOWN_FLAGS 0x01

// version byte + flags byte, present on every v3 payload
#define EFFECTS_V3_HEADER_LEN 2

// bytes a compressed payload adds before the frame: two u32 lengths and the
// u32 checksum
#define EFFECTS_V3_COMPRESSED_PREFIX 12

// zstd level for effects payloads.
//
// Level 1: the payload is built on the write thread under the lock and the
// corpus is repetitive, so the cheapest level gets most of the ratio. It is
// also what the format specifies and what the Rust encoder emits.
//
// Its ratio is alignment-sensitive - the same body measured 10,399 bytes at
// one alignment and ~31,100 at the other eight - so a measured ratio is a
// property of the payload, not of the format.
#define EFFECTS_V3_COMPRESSION_LEVEL 1

// is compressing worth it?
//
// The frame replaces the record stream but the twelve byte prefix rides along,
// so the frame must be STRICTLY more than twelve bytes smaller. At break-even
// the compressed form is the same size and costs both ends a zstd pass, so it
// stays uncompressed.
static inline bool EffectsV3_CompressionWorthIt
(
	size_t frame_len,   // compressed frame length
	size_t records_len  // record stream it would replace
) {
	return frame_len + EFFECTS_V3_COMPRESSED_PREFIX < records_len ;
}

// compress a finished v3 payload, if that makes it smaller
//
// '*payload' is the COMPLETE payload including its two byte header, as
// produced by the encoder. On success '*payload' is replaced by a freshly
// allocated compressed payload, '*len' is updated, the old buffer is freed and
// true is returned. On any refusal the payload is left exactly as it was and
// false is returned.
//
// Refuses, rather than failing the write, when:
//
//   * 'min_bytes' is 0 (compression disabled) or the record stream is smaller
//     than it
//   * the payload already has FLAG_COMPRESSED set. Compressing twice produces
//     a payload nothing can read, since a reader inflates once and the second
//     pass buried the first one's prefix in the frame. A refusal rather than
//     an ASSERT: refusing IS the safety property, and ASSERT compiles to
//     nothing in release builds (RG.h), which is where it would matter
//   * zstd fails - the uncompressed payload is still correct, so this is not
//     a reason to fail a write
//   * the frame would not be strictly smaller (see CompressionWorthIt)
//
// MUST be the last thing that touches the bytes: it rewrites everything after
// the header.
bool EffectsV3_MaybeCompress
(
	char **payload,   // [input/output] complete payload, replaced on success
	size_t *len,      // [input/output] payload length
	size_t min_bytes  // smallest record stream worth compressing, 0 disables
);

// which check refused a compressed payload
//
// EffectsV3Status collapses all of these to MALFORMED, which is right on the
// wire and wrong in a log: "bytes follow the frame" and "checksum disagrees"
// send an operator to different places. Rust carries them as distinct
// variants. Reported through an optional out-parameter rather than by
// widening EffectsV3Status, which is a shared contract.
typedef enum {
	EFFECTS_V3_COMPRESS_OK = 0,          // no fault
	EFFECTS_V3_COMPRESS_SHORT_PREFIX,    // ended inside the 12 byte prefix
	EFFECTS_V3_COMPRESS_SHORT_FRAME,     // compressed_length outruns the buffer
	EFFECTS_V3_COMPRESS_TRAILING,        // bytes after the frame
	EFFECTS_V3_COMPRESS_BAD_FRAME,       // zstd refused it
	EFFECTS_V3_COMPRESS_LENGTH_MISMATCH, // expanded to != uncompressed_length
	EFFECTS_V3_COMPRESS_CHECKSUM,        // plaintext CRC-32 disagrees
	EFFECTS_V3_COMPRESS_NO_MEMORY,       // could not allocate uncompressed_length
} EffectsV3CompressFault;

// human readable form of a fault, for logs and test failures
const char *EffectsV3CompressFault_ToString
(
	EffectsV3CompressFault fault
);

// inflate a compressed v3 payload
//
// 'body' points at the TWELVE BYTE PREFIX, just past the version and flags
// bytes; 'body_len' runs from there to the end of the buffer. On
// EFFECTS_V3_OK the caller owns '*plain' and must rm_free it; otherwise
// '*plain' is NULL and nothing is left allocated.
//
// The plaintext is a v3 record stream WITHOUT a header - what the
// uncompressed form carries after its two header bytes.
EffectsV3Status EffectsV3_OpenCompressed
(
	const char *body,              // compressed prefix followed by the frame
	size_t body_len,               // bytes available from 'body'
	char **plain,                  // [output] inflated record stream
	size_t *plain_len,             // [output] its length
	EffectsV3CompressFault *fault  // [output, optional] which check fired
);
