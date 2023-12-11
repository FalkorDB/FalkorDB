/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "ws.h"
#include "endian.h"
#include "util/rmalloc.h"
#include <string.h>
#include <openssl/bio.h>
#include <openssl/evp.h>
#include <openssl/sha.h>

#define WS_GUID "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

typedef struct ws_upgrade_request {
	char *request_uri;
	char *host;
	char *upgrade;
	char *connection;
	char *sec_ws_key;
	char *origin;
	char *sec_ws_protocol;
	int sec_ws_version;
} ws_upgrade_request_t;

// parse the headers of a websocket handshake
// return true if the request is a websocket handshake
static bool parse_headers
(
	buffer_index_t *request,       // the request buffer
	ws_upgrade_request_t *headers  // the headers value
) {
	ASSERT(request != NULL);
	ASSERT(headers != NULL);

	uint32_t size = buffer_index_diff(&request->buf->write, request);
	char *data = rm_malloc(size + 1);
	buffer_index_read(request, data, size);
	data[size] = '\0';
	bool is_ws = false;
	uint start_line = 0;
	uint end_line = strchr(data, '\r') - data;
	while(start_line < size) {
		if(strncmp(data + start_line, "GET ", 4) == 0) {
			headers->request_uri = rm_strndup(data + start_line + 4, end_line - start_line - 4);
		} else if(strncmp(data + start_line, "Host: ", 6) == 0) {
			headers->host = rm_strndup(data + start_line + 6, end_line - start_line - 6);
		} else if(strncmp(data + start_line, "Upgrade: ", 9) == 0) {
			headers->upgrade = rm_strndup(data + start_line + 9, end_line - start_line - 9);
		} else if(strncmp(data + start_line, "Connection: ", 12) == 0) {
			headers->connection = rm_strndup(data + start_line + 12, end_line - start_line - 12);
		} else if(strncmp(data + start_line, "Sec-WebSocket-Key: ", 19) == 0) {
			headers->sec_ws_key = rm_strndup(data + start_line + 19, end_line - start_line - 19);
			is_ws = true;
		} else if(strncmp(data + start_line, "Origin: ", 8) == 0) {
			headers->origin = rm_strndup(data + start_line + 8, end_line - start_line - 8);
		} else if(strncmp(data + start_line, "Sec-WebSocket-Protocol: ", 24) == 0) {
			headers->sec_ws_protocol = rm_strndup(data + start_line + 24, end_line - start_line - 24);
		} else if(strncmp(data + start_line, "Sec-WebSocket-Version: ", 23) == 0) {
			headers->sec_ws_version = atoi(data + start_line + 23);
		}
		start_line = end_line + 2;
		end_line = strchr(data + start_line, '\r') - data;
	}
	rm_free(data);
	return is_ws;
}

static bool validate_headers
(
	ws_upgrade_request_t *headers  // the headers value
) {
	ASSERT(headers != NULL);

	if(headers->request_uri == NULL || strcmp(headers->request_uri, "/ HTTP/1.1") != 0) {
		return false;
	}
	if(headers->host == NULL) {
		return false;
	}
	if(headers->upgrade == NULL || strcmp(headers->upgrade, "websocket") != 0) {
		return false;
	}
	if(headers->connection == NULL || strcmp(headers->connection, "keep-alive, Upgrade") != 0) {
		return false;
	}
	if(headers->sec_ws_key == NULL) {
		return false;
	}
	if(headers->origin == NULL) {
		return false;
	}
	if(headers->sec_ws_version != 13) {
		return false;
	}
	return true;
}

static void free_headers
(
	ws_upgrade_request_t *headers  // the headers value
) {
	ASSERT(headers != NULL);

	rm_free(headers->request_uri);
	rm_free(headers->host);
	rm_free(headers->upgrade);
	rm_free(headers->connection);
	rm_free(headers->sec_ws_key);
	rm_free(headers->origin);
	rm_free(headers->sec_ws_protocol);
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

	ws_upgrade_request_t headers = {0};
	if(!parse_headers(request, &headers) || !validate_headers(&headers)) {
		return false;
	}

	// hash the sec-websocket-key header and ws guid
	unsigned char hash[SHA_DIGEST_LENGTH];
	EVP_MD_CTX *mdctx = EVP_MD_CTX_new();
	EVP_DigestInit_ex(mdctx, EVP_sha1(), NULL);
	EVP_DigestUpdate(mdctx, headers.sec_ws_key, strlen(headers.sec_ws_key));
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
	buffer_write(response, "HTTP/1.1 101 Switching Protocols\r\n", 34);
	buffer_write(response, "Upgrade: websocket\r\n", 20);
	buffer_write(response, "Connection: Upgrade\r\n", 21);
	buffer_write(response, "Sec-WebSocket-Accept: ", 22);
	buffer_write(response, encoded, len);
	buffer_write(response, "\r\n\r\n", 4);
	BIO_free_all(b64);
	free_headers(&headers);
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
		buffer_apply_mask(*buf, masking_key, payload_len);
	}
	return payload_len;
}

// write an empty websocket frame header
uint64_t ws_write_empty_header
(
	buffer_index_t *buf  // the buffer to write to
) {
	ASSERT(buf != NULL);

	buffer_write_uint32(buf, 0x00000000);
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
		buffer_write_uint32(&msg, htonl(0x827E0000 + n));
	} else {
		buffer_write_uint16(buf, 0x0000);
		msg = *buf;
		buffer_write_uint8(&msg, 0x82);
		buffer_write_uint8(&msg, n);
	}
}
