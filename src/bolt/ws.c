/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "ws.h"
#include "endian.h"
#include <string.h>
#include <openssl/bio.h>
#include <openssl/evp.h>
#include <openssl/sha.h>

#define WS_GUID "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

// parse the headers of a websocket handshake
// return true if the request is a websocket handshake
bool parse_headers
(
	buffer_index_t *request,  // the request buffer
	char **sec_ws_key         // the sec-websocket-key header
) {
	ASSERT(request != NULL);
	ASSERT(sec_ws_key != NULL);

	char *data = buffer_index_read(request, 0);
	char *line = strtok(data, "\r\n");
	while (line != NULL) {
		if (strncmp(line, "Sec-WebSocket-Key: ", 19) == 0) {
			*sec_ws_key = line + 19;
			return true;
		}
		line = strtok(NULL, "\r\n");
	}
	return false;
}

// check if the request is a websocket handshake
// write the response to the response buffer
bool ws_handshake
(
	buffer_index_t *request,  // the request buffer
	buffer_index_t *response  // the response buffer
) {
	ASSERT(request != NULL);
	ASSERT(response != NULL);

	char *sec_ws_key;
	if(!parse_headers(request, &sec_ws_key)) {
		return false;
	}

	// hash the sec-websocket-key header and ws guid
	unsigned char hash[SHA_DIGEST_LENGTH];
	EVP_MD_CTX *mdctx = EVP_MD_CTX_new();
	EVP_DigestInit_ex(mdctx, EVP_sha1(), NULL);
	EVP_DigestUpdate(mdctx, sec_ws_key, strlen(sec_ws_key));
	EVP_DigestUpdate(mdctx, WS_GUID, strlen(WS_GUID));
	EVP_DigestFinal_ex(mdctx, hash, NULL);
	EVP_MD_CTX_free(mdctx);

	// base64 encode the hash
	BIO *b64 = BIO_new(BIO_f_base64());
	BIO_set_flags(b64, BIO_FLAGS_BASE64_NO_NL);
	BIO *bmem = BIO_new(BIO_s_mem());
	b64 = BIO_push(b64, bmem);
	BIO_write(b64, hash, SHA_DIGEST_LENGTH);
	BIO_flush(b64);
	char encoded[29];
	int len = BIO_read(bmem, encoded, 28);
	encoded[len] = '\0';

	// write the response
	buffer_write(response, "HTTP/1.1 101 Switching Protocols\r\n", 34);
	buffer_write(response, "Upgrade: websocket\r\n", 20);
	buffer_write(response, "Connection: Upgrade\r\n", 21);
	buffer_write(response, "Sec-WebSocket-Accept: ", 22);
	buffer_write(response, encoded, len);
	buffer_write(response, "\r\n\r\n", 4);
	BIO_free_all(b64);
	return true;
}

// read a websocket frame header returning the payload length
uint64_t ws_read_frame
(
	buffer_index_t *buf  // the buffer to read from
) {
	ASSERT(buf != NULL);

	// https://www.rfc-editor.org/rfc/rfc6455#section-5.2
	//   0                   1                   2                   3
	//   0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
	//  +-+-+-+-+-------+-+-------------+-------------------------------+
	//  |F|R|R|R| opcode|M| Payload len |    Extended payload length    |
	//  |I|S|S|S|  (4)  |A|     (7)     |             (16/64)           |
	//  |N|V|V|V|       |S|             |   (if payload len==126/127)   |
	//  | |1|2|3|       |K|             |                               |
	//  +-+-+-+-+-------+-+-------------+ - - - - - - - - - - - - - - - +
	//  |     Extended payload length continued, if payload len == 127  |
	//  + - - - - - - - - - - - - - - - +-------------------------------+
	//  |                               |Masking-key, if MASK set to 1  |
	//  +-------------------------------+-------------------------------+
	//  | Masking-key (continued)       |          Payload Data         |
	//  +-------------------------------- - - - - - - - - - - - - - - - +
	//  :                     Payload Data continued ...                :
	//  + - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - +
	//  |                     Payload Data continued ...                |
	//  +---------------------------------------------------------------+

	uint16_t frame_header = ntohs(buffer_read_uint16(buf));
	bool fin = frame_header >> 15;
	ASSERT(fin && "Fragmented frames are not supported");
	uint8_t rsv123 = (frame_header >> 12) & 0x07;
	ASSERT(rsv123 == 0 && "Reserved bits are not supported");
	uint8_t opcode = (frame_header >> 8) & 0x0F;
	ASSERT(opcode == 0x02 || opcode == 0x08 && "Only binary frames are supported");
	uint64_t payload_len = frame_header & 0x7F;
	if(payload_len == 126) {
		payload_len = ntohs(buffer_read_uint16(buf));
	} else if(payload_len == 127) {
		payload_len = ntohll(buffer_read_uint64(buf));
	}
	bool mask = (frame_header >> 7) & 0x1;
	if(mask) {
		uint32_t masking_key = buffer_read_uint32(buf);
		char *payload = buffer_index_read(buf, 0);
		for(int i = 0; i < payload_len; i++) {
			payload[i] ^= ((char*)&masking_key)[i % 4];
		}
	}
	return payload_len;
}

// write a websocket frame header
uint64_t ws_write_empty_header
(
	buffer_index_t *buf  // the buffer to write to
) {
	buffer_write_uint32(buf, 0x00000000);
}
