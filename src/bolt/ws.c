/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "ws.h"
#include "endian.h"
#include "rax/rax.h"
#include "util/rmalloc.h"
#include <string.h>
#include <openssl/bio.h>
#include <openssl/evp.h>
#include <openssl/sha.h>

#define WS_GUID "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

// parse the headers of a websocket handshake
// return true if the request is a websocket handshake
static bool parse_headers
(
	buffer_index_t *request, // the request buffer
	rax *headers             // the headers value
) {
	ASSERT(request != NULL);
	ASSERT(headers != NULL);

	char request_line[16];
	if(!buffer_read_n(request, request_line, 16)) {
		return false;
	}
	if(strncmp(request_line, "GET / HTTP/1.1\r\n", 16) != 0) {
		return false;
	}
	while(buffer_index_length(request) > 2) {
		char *field = NULL;
		if(!buffer_index_read_until(request, ':', &field)) {
			return false;
		}
		buffer_index_advance(request, 2);
		char *value = NULL;
		if(!buffer_index_read_until(request, '\r', &value)) {
			rm_free(field);
			return false;
		}
		buffer_index_advance(request, 2);
		raxInsert(headers, (unsigned char *)field, strlen(field), (void *)value, NULL);
		rm_free(field);
	}
	return buffer_index_advance(request, 2);
}

static bool validate_headers
(
	rax *headers  // the headers value
) {
	ASSERT(headers != NULL);

	void *v = NULL;
	char *k = NULL;

	// locate host
	k = "Host";
	if(raxFind(headers, (unsigned char *)k, 4) == raxNotFound) return false;

	// make sure upgrade is set to websocket
	k = "Upgrade";
	v = raxFind(headers, (unsigned char *)k, 7);
	if(v == raxNotFound || strcmp((char *)v, "websocket") != 0) {
		return false;
	}

	// locate connection
	k = "Connection";
	v = raxFind(headers, (unsigned char *)k, 10);
	if(v == raxNotFound) return false;

	// make sure connection is set to upgrade
	char *i = v;
	bool is_upgrade = false;
	while(i != NULL && strlen(i) > 0) {
		is_upgrade = (strncmp(i, "Upgrade", 7) == 0);
		if(is_upgrade) break;
		i = strchr(i, ',') + 2;
	}

	if(!is_upgrade) return false;

	// locate Sec-WebSocket-Key
	k = "Sec-WebSocket-Key";
	if(raxFind(headers, (unsigned char *)k, 17) == raxNotFound) return false;

	// make sure Sec-WebSocket-Version is set to 13
	k = "Sec-WebSocket-Version";
	v = raxFind(headers, (unsigned char *)k, 21);
	if(v == raxNotFound || strcmp((char *)v, "13") != 0) return false;

	// locate Origin
	k = "Origin";
	if(raxFind(headers, (unsigned char*)k, 6) == raxNotFound) return false;

	return true;
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

	rax *headers = raxNew();
	if(!parse_headers(request, headers) || !validate_headers(headers)) {
		raxFreeWithCallback(headers, rm_free);
		return false;
	}

	// hash the sec-websocket-key header and ws guid
	unsigned char hash[SHA_DIGEST_LENGTH];
	EVP_MD_CTX *mdctx = EVP_MD_CTX_new();
	EVP_DigestInit_ex(mdctx, EVP_sha1(), NULL);
	char *key = "Sec-WebSocket-Key";
	char *sec_ws_key = raxFind(headers, (unsigned char*)key, 17);
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
	ASSERT(len == 28);
	encoded[len] = '\0';

	// write the response
	buffer_write_n(response, "HTTP/1.1 101 Switching Protocols\r\n", 34);
	buffer_write_n(response, "Upgrade: websocket\r\n", 20);
	buffer_write_n(response, "Connection: Upgrade\r\n", 21);
	buffer_write_n(response, "Sec-WebSocket-Accept: ", 22);
	buffer_write_n(response, encoded, len);
	buffer_write_n(response, "\r\n\r\n", 4);
	BIO_free_all(b64);
	raxFreeWithCallback(headers, rm_free);
	return true;
}

// read a websocket frame header returning the payload length
bool ws_read_frame
(
	buffer_index_t *buf,   // the buffer to read from
	uint64_t *payload_len  // the payload length
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

	uint16_t frame_header;
	if(!buffer_read(buf, &frame_header)) {
		return false;
	}
	frame_header = ntohs(frame_header);
	bool fin = frame_header >> 15;
	ASSERT(fin && "Fragmented frames are not supported");
	uint8_t rsv123 = (frame_header >> 12) & 0x07;
	ASSERT(rsv123 == 0 && "Reserved bits are not supported");
	uint8_t opcode = (frame_header >> 8) & 0x0F;
	ASSERT(opcode == 0x02 || opcode == 0x08 && "Only binary frames are supported");
	if((frame_header & 0x7F) == 126) {
		uint16_t _payload_len;
		if(!buffer_read(buf, &_payload_len)) {
			return false;
		}
		*payload_len = ntohs(_payload_len);
	} else if((frame_header & 0x7F) == 127) {
		uint64_t _payload_len;
		if(!buffer_read(buf, &_payload_len)) {
			return false;
		}
		*payload_len = ntohll(_payload_len);
	} else {
		*payload_len = frame_header & 0x7F;
	}
	bool mask = (frame_header >> 7) & 0x1;
	if(mask) {
		uint32_t masking_key;
		if(!buffer_read(buf, &masking_key)) {
			return false;
		}
		buffer_apply_mask(*buf, masking_key, *payload_len);
	}
	return true;
}

// write an empty websocket frame header
void ws_write_empty_header
(
	buffer_index_t *buf  // the buffer to write to
) {
	ASSERT(buf != NULL);

	buffer_write_uint32_t(buf, 0x00000000);
}

// write a websocket frame header
void ws_write_frame_header
(
	buffer_index_t *buf,  // the buffer to write to
	uint64_t n            // the payload length
) {
	ASSERT(buf != NULL);

	buffer_index_t msg = *buf;
	if(n > 125) {
		buffer_write_uint32_t(&msg, htonl(0x827E0000 + n));
	} else {
		buffer_write_uint16_t(buf, 0x0000);
		msg = *buf;
		buffer_write_uint8_t(&msg, 0x82);
		buffer_write_uint8_t(&msg, n);
	}
}

