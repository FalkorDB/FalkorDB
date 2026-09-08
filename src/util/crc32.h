/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

// CRC-32/ISO-HDLC, also known as the zlib or gzip CRC-32
//
//   polynomial  0x04C11DB7, reflected as 0xEDB88320
//   init        0xFFFFFFFF
//   reflect     input and output
//   final xor   0xFFFFFFFF
//   check       CRC32("123456789") == 0xCBF43926
//
// This is the algorithm the effects v3 wire format uses for the checksum a
// compressed payload carries, taken over the PLAINTEXT rather than the zstd
// frame - the plaintext is the bytes records are parsed from, and it stays
// reproducible across zstd versions, which may frame the same input
// differently.
//
// It is eight lines and a table rather than a link against something that
// already has one. There are two such functions in the build already -
// mz_crc32 from RediSearch's vendored miniz, and CRoaring's - and reaching
// into another submodule's internals for a wire-format checksum is a
// dependency this file exists to avoid. It also would not build in a worktree
// where that submodule is not populated.
uint32_t CRC32
(
	const void *data,  // bytes to checksum
	size_t n           // number of bytes
);
