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
// The header is never compressed, so a reader always knows what it holds
// before committing to decode anything.
//
// Three consequences, each of which is a rule rather than a preference:
//
//   * the compressed-only prefix is TWELVE bytes, so compression must save
//     more than twelve to be worth doing. Getting this wrong does not corrupt
//     anything, it just makes payloads bigger.
//   * `compressed_length` makes the payload self-delimiting, so a reader takes
//     exactly that many bytes rather than reading to the end of the buffer,
//     and TRAILING BYTES ARE A DECODE ERROR rather than tolerated padding.
//   * `uncompressed_length` is the decompress allocation CEILING, applied
//     before the allocation, and then cross-checked afterwards. Both, because
//     they fail on different inputs - see EffectsV3_OpenCompressed.
//
// The checksum is over the plaintext, not the frame: those are the bytes
// records are parsed from, and it stays reproducible across zstd versions,
// which may frame the same input differently.
//
// Compression is DEFAULT OFF, and `EFFECTS_COMPRESSION` is a byte threshold
// rather than a boolean - the smallest record stream worth compressing, with
// 0 disabling it. It is a bandwidth trade, not a CPU one.
//
//------------------------------------------------------------------------------

// flags bit 0: the records are a zstd frame rather than a record stream
#define EFFECTS_V3_FLAG_COMPRESSED 0x01

// every flag bit this build understands. A reader meeting anything outside
// this mask REFUSES the buffer rather than masking it off - an old node
// meeting a future payload must fail loudly, because decoding the records
// anyway would apply a prefix of something whose shape it does not know
#define EFFECTS_V3_KNOWN_FLAGS 0x01

// version byte + flags byte, present on every v3 payload
#define EFFECTS_V3_HEADER_LEN 2

// bytes a compressed payload adds before the frame: two u32 lengths and the
// u32 checksum. Compression has to save MORE than this to be worth doing
#define EFFECTS_V3_COMPRESSED_PREFIX 12

// zstd level for effects payloads.
//
// Level 1. The payload is built on the write thread while it holds the lock,
// and the corpus is highly repetitive, so the cheapest level already gets most
// of the ratio.
//
// Level 1's ratio here is a coin flip on alignment, which is worth knowing
// before quoting a number: its fast match-finder phase-locks onto the record
// period, so shifting a payload by one byte swings the output. Measured on the
// Rust side over 10,000 CREATE_NODE value rows, the same body compresses to
// 10,399 bytes at one alignment and ~31,100 at the other eight. Level 3 gives
// ~27,250 at every alignment - worse than level 1's lucky case, better than
// its common one, and stable, which is what a wire format actually wants.
//
// Left at 1 because that is what the format specifies and what the Rust
// encoder emits; raising it is a CPU trade nobody has measured on the write
// thread yet.
#define EFFECTS_V3_COMPRESSION_LEVEL 1

// is compressing worth it?
//
// The frame replaces the record stream, but the two declared lengths and the
// checksum ride along, so a frame has to be MORE than twelve bytes smaller
// than the records it replaces. Comparing against 0, or against 8 as an
// earlier reading of the format did, silently turns a payload that saves nine
// bytes into a payload that grew.
//
// Strictly smaller, so a payload that exactly breaks even stays uncompressed:
// at break-even the compressed form is the same size and strictly worse, since
// it costs both ends a zstd pass and makes the bytes unreadable to anything
// that cannot inflate them.
//
// Named rather than written inline at its one call site, because the boundary
// is the part that gets it wrong and a named predicate can be pinned by a test
// without constructing a payload that happens to save exactly twelve bytes.
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
//     a payload nothing can read: the second pass would swallow the first
//     one's lengths and checksum as if they were records, and a reader
//     inflates once. This must be called exactly once, on a finished payload,
//     and "exactly once" is an easy thing for a caller to get wrong when a
//     query commits more than once. Deliberately not an ASSERT: refusing IS
//     the safety property, and ASSERT compiles to nothing in release builds
//     (RG.h), which is precisely where it would matter
//   * zstd fails. The uncompressed payload is still correct, so a compression
//     failure is not a reason to fail a write
//   * the frame plus the twelve byte prefix would not be strictly smaller than
//     the record stream it replaces
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
// EffectsV3Status collapses every one of these to MALFORMED, which is the
// right thing on the wire - a refusal is a refusal and the caller treats it as
// divergence either way. It is the wrong thing in a log: "bytes follow the
// compressed frame" and "plaintext checksum disagrees" send an operator to
// completely different places. The Rust reader carries these as distinct error
// variants, so collapsing them in C would be a parity gap in observability
// rather than in behaviour.
//
// Reported through an optional out-parameter rather than by widening
// EffectsV3Status, because that enum is the organizer's shared contract and
// this detail is local to compression.
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
// 'body' points at the TWELVE BYTE PREFIX, i.e. just past the version and
// flags bytes, and 'body_len' is everything from there to the end of the
// buffer. On EFFECTS_V3_OK the caller owns '*plain' and must rm_free it; on
// anything else '*plain' is NULL and nothing is left allocated.
//
// The plaintext it returns is a v3 record stream WITHOUT a header - it is what
// the uncompressed form carries after its two header bytes.
EffectsV3Status EffectsV3_OpenCompressed
(
	const char *body,              // compressed prefix followed by the frame
	size_t body_len,               // bytes available from 'body'
	char **plain,                  // [output] inflated record stream
	size_t *plain_len,             // [output] its length
	EffectsV3CompressFault *fault  // [output, optional] which check fired
);
