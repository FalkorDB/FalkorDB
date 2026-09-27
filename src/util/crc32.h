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
// Used for the checksum a compressed effects v3 payload carries, taken over
// the PLAINTEXT rather than the zstd frame.
//
// Written out rather than linked against RediSearch's mz_crc32 or CRoaring's:
// reaching into another submodule's internals for a wire-format checksum is a
// dependency this file exists to avoid, and it would not build where that
// submodule is unpopulated.
uint32_t CRC32
(
	const void *data,  // bytes to checksum
	size_t n           // number of bytes
);
