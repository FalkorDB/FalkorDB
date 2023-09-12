/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "bolt.h"
#include "bolt_client.h"
#include "util/rmalloc.h"
#include <arpa/inet.h>

bolt_client_t *bolt_client_new
(
	socket_t socket,
	RedisModuleCtx *ctx,
	RedisModuleEventLoopFunc on_write
) {
	bolt_client_t *client = rm_malloc(sizeof(bolt_client_t));
	client->socket = socket;
	client->state = BS_NEGOTIATION;
	client->ctx = ctx;
	client->on_write = on_write;
	client->nwrite = 0;
	client->nread = 0;
	client->processing = false;
	client->shutdown = false;
	client->reset = false;
	client->last_read_index = 0;
	client->last_write_index = 2;
	return client;
}

void bolt_change_negotiation_state
(
	bolt_client_t *client,
	bolt_structure_type request_type,
	bolt_structure_type response_type
) {
	ASSERT(client->state == BS_NEGOTIATION && request_type == BST_HELLO);

	switch (response_type)
	{
		case BST_SUCCESS:
			client->state = BS_AUTHENTICATION;
			break;
		case BST_FAILURE:
			client->state = BS_DEFUNCT;
			break;
		default:
			ASSERT(false);
	}
}

void bolt_change_authentication_state
(
	bolt_client_t *client,
	bolt_structure_type request_type,
	bolt_structure_type response_type
) {
	ASSERT(client->state == BS_AUTHENTICATION && request_type == BST_LOGON);

	switch (response_type)
	{
		case BST_SUCCESS:
			client->state = BS_READY;
			break;
		case BST_FAILURE:
			client->state = BS_DEFUNCT;
			break;
		default:
			ASSERT(false);
	}
}

void bolt_change_ready_state
(
	bolt_client_t *client,
	bolt_structure_type request_type,
	bolt_structure_type response_type
) {
	ASSERT(client->state == BS_READY);

	switch (request_type)
	{
		case BST_LOGOFF:
			switch (response_type)
			{
				case BST_SUCCESS:
					client->state = BS_AUTHENTICATION;
					break;
				case BST_FAILURE:
					client->state = BS_FAILED;
					break;
				default:
					ASSERT(false);
			}
			break;
		case BST_RUN:
			switch (response_type)
			{
				case BST_SUCCESS:
					client->state = BS_STREAMING;
					break;
				case BST_FAILURE:
					client->state = BS_FAILED;
					break;
				default:
					ASSERT(false);
			}
			break;
		case BST_BEGIN:
			switch (response_type)
			{
				case BST_SUCCESS:
					client->state = BS_TX_READY;
					break;
				case BST_FAILURE:
					client->state = BS_FAILED;
					break;
				default:
					ASSERT(false);
			}
			break;
		case BST_ROUTE:
			switch (response_type)
			{
				case BST_SUCCESS:
					client->state = BS_READY;
					break;
				default:
					ASSERT(false);
			}
			break;
		case BST_RESET:
			client->state = BS_READY;
			break;
		case BST_GOODBYE:
			client->state = BS_DEFUNCT;
			break;
		default:
			ASSERT(false);
	}
}

void bolt_change_streaming_state
(
	bolt_client_t *client,
	bolt_structure_type request_type,
	bolt_structure_type response_type
) {
	ASSERT(client->state == BS_STREAMING);

	switch (request_type)
	{
		case BST_PULL:
			switch (response_type)
			{
				case BST_SUCCESS:
					client->state = BS_READY;
					break;
				case BST_FAILURE:
					client->state = BS_FAILED;
					break;
				default:
					ASSERT(false);
			}
			break;
		case BST_DISCARD:
			switch (response_type)
			{
				case BST_SUCCESS:
					client->state = BS_READY;
					break;
				case BST_FAILURE:
					client->state = BS_FAILED;
					break;
				default:
					ASSERT(false);
			}
			break;
		case BST_RESET:
			client->state = BS_READY;
			break;
		case BST_GOODBYE:
			client->state = BS_DEFUNCT;
			break;
		default:
			ASSERT(false);
	}
}

void bolt_change_txready_state
(
	bolt_client_t *client,
	bolt_structure_type request_type,
	bolt_structure_type response_type
) {
	ASSERT(client->state == BS_TX_READY);

	switch (request_type)
	{
		case BST_RUN:
			switch (response_type)
			{
				case BST_SUCCESS:
					client->state = BS_TX_STREAMING;
					break;
				case BST_FAILURE:
					client->state = BS_FAILED;
					break;
				default:
					ASSERT(false);
			}
			break;
		case BST_COMMIT:
			switch (response_type)
			{
				case BST_SUCCESS:
					client->state = BS_READY;
					break;
				case BST_FAILURE:
					client->state = BS_FAILED;
					break;
				default:
					ASSERT(false);
			}
			break;
		case BST_ROLLBACK:
			switch (response_type)
			{
				case BST_SUCCESS:
					client->state = BS_READY;
					break;
				case BST_FAILURE:
					client->state = BS_FAILED;
					break;
				default:
					ASSERT(false);
			}
			break;
		case BST_RESET:
			client->state = BS_READY;
			break;
		case BST_GOODBYE:
			client->state = BS_DEFUNCT;
			break;
		default:
			ASSERT(false);
	}
}

void bolt_change_txstreaming_state
(
	bolt_client_t *client,
	bolt_structure_type request_type,
	bolt_structure_type response_type
) {
	ASSERT(client->state == BS_TX_STREAMING);

	switch (request_type)
	{
		case BST_RUN:
			switch (response_type)
			{
				case BST_SUCCESS:
					client->state = BS_TX_STREAMING;
					break;
				case BST_FAILURE:
					client->state = BS_FAILED;
					break;
				default:
					ASSERT(false);
			}
			break;
		case BST_PULL:
			switch (response_type)
			{
				case BST_SUCCESS:
					client->state = BS_TX_STREAMING;
					break;
				case BST_FAILURE:
					client->state = BS_FAILED;
					break;
				default:
					ASSERT(false);
			}
			break;
		case BST_COMMIT:
			switch (response_type)
			{
				case BST_SUCCESS:
					client->state = BS_READY;
					break;
				case BST_FAILURE:
					client->state = BS_FAILED;
					break;
				default:
					ASSERT(false);
			}
			break;
		case BST_DISCARD:
			switch (response_type)
			{
				case BST_SUCCESS:
					client->state = BS_TX_READY;
					break;
				case BST_FAILURE:
					client->state = BS_FAILED;
					break;
				default:
					ASSERT(false);
			}
			break;
		case BST_RESET:
			client->state = BS_READY;
			break;
		case BST_GOODBYE:
			client->state = BS_DEFUNCT;
			break;
		default:
			ASSERT(false);
	}
}

void bolt_change_failed_state
(
	bolt_client_t *client,
	bolt_structure_type request_type,
	bolt_structure_type response_type
) {
	ASSERT(client->state == BS_FAILED);

	switch (request_type)
	{
		case BST_RUN:
			switch (response_type)
			{
				case BST_IGNORED:
					client->state = BS_FAILED;
					break;
				default:
					ASSERT(false);
			}
			break;
		case BST_PULL:
			switch (response_type)
			{
				case BST_IGNORED:
					client->state = BS_FAILED;
					break;
				default:
					ASSERT(false);
			}
			break;
		case BST_DISCARD:
			switch (response_type)
			{
				case BST_IGNORED:
					client->state = BS_FAILED;
					break;
				default:
					ASSERT(false);
			}
			break;
		case BST_RESET:
			client->state = BS_READY;
			break;
		case BST_GOODBYE:
			client->state = BS_DEFUNCT;
			break;
		default:
			ASSERT(false);
	}
}

void bolt_change_interrupted_state
(
	bolt_client_t *client,
	bolt_structure_type request_type,
	bolt_structure_type response_type
) {
	ASSERT(client->state == BS_INTERRUPTED);

	switch (request_type)
	{
		case BST_RUN:
			switch (response_type)
			{
				case BST_IGNORED:
					client->state = BS_FAILED;
					break;
				default:
					ASSERT(false);
			}
			break;
		case BST_PULL:
			switch (response_type)
			{
				case BST_IGNORED:
					client->state = BS_FAILED;
					break;
				default:
					ASSERT(false);
			}
			break;
		case BST_DISCARD:
			switch (response_type)
			{
				case BST_IGNORED:
					client->state = BS_FAILED;
					break;
				default:
					ASSERT(false);
			}
			break;
		case BST_BEGIN:
			switch (response_type)
			{
				case BST_IGNORED:
					client->state = BS_FAILED;
					break;
				default:
					ASSERT(false);
			}
			break;
		case BST_COMMIT:
			switch (response_type)
			{
				case BST_IGNORED:
					client->state = BS_FAILED;
					break;
				default:
					ASSERT(false);
			}
			break;
		case BST_ROLLBACK:
			switch (response_type)
			{
				case BST_IGNORED:
					client->state = BS_FAILED;
					break;
				default:
					ASSERT(false);
			}
			break;
		case BST_RESET:
			switch (response_type)
			{
				case BST_SUCCESS:
					client->state = BS_READY;
					break;
				case BST_FAILURE:
					client->state = BS_DEFUNCT;
					break;
				default:
					ASSERT(false);
			}
			break;
		case BST_GOODBYE:
			client->state = BS_DEFUNCT;
			break;
		default:
			ASSERT(false);
	}
}

void bolt_change_client_state
(
	bolt_client_t *client,
	bolt_structure_type request_type,
	bolt_structure_type response_type
) {
	if(response_type == BST_RECORD) {
		return;
	}
	switch (client->state)
	{
		case BS_NEGOTIATION:
			bolt_change_negotiation_state(client, request_type, response_type);
			break;
		case BS_AUTHENTICATION:
			bolt_change_authentication_state(client, request_type, response_type);
			break;
		case BS_READY:
			bolt_change_ready_state(client, request_type, response_type);
			break;
		case BS_STREAMING:
			bolt_change_streaming_state(client, request_type, response_type);
			break;
		case BS_TX_READY:
			bolt_change_txready_state(client, request_type, response_type);
			break;
		case BS_TX_STREAMING:
			bolt_change_txstreaming_state(client, request_type, response_type);
			break;
		case BS_FAILED:
			bolt_change_failed_state(client, request_type, response_type);
			break;
		case BS_INTERRUPTED:
			bolt_change_interrupted_state(client, request_type, response_type);
			break;
		default:
			ASSERT(false);
			break;
	}
}

void bolt_client_reply_for
(
	bolt_client_t *client,
	bolt_structure_type request_type,
	bolt_structure_type response_type,
	uint32_t size
) {
	bolt_reply_structure(client, response_type, size);
	bolt_change_client_state(client, request_type, response_type);
}

void bolt_client_end_message
(
	bolt_client_t *client
) {
	uint16_t n = client->last_write_index - client->nwrite - 2;
	ASSERT(n > 2);
	
	*(u_int16_t *)(client->write_buffer + client->nwrite) = htons(n);
	client->write_buffer[client->last_write_index++] = 0x00;
	client->write_buffer[client->last_write_index++] = 0x00;
	client->nwrite = client->last_write_index;
	client->last_write_index += 2;
}

void bolt_client_finish_write
(
	bolt_client_t *client
) {
	RedisModule_EventLoopAdd(client->socket, REDISMODULE_EVENTLOOP_WRITABLE, client->on_write, client);
}

void bolt_client_send
(
	bolt_client_t *client
) {
	if(client->reset) {
		if(client->state != BS_FAILED) {
			client->last_write_index = 2;
			bolt_reply_structure(client, BST_SUCCESS, 1);
			bolt_reply_map(client, 0);
			uint16_t n = client->last_write_index - 2;
			*(u_int16_t *)client->write_buffer = htons(n);
			client->write_buffer[n + 2] = 0x00;
			client->write_buffer[n + 3] = 0x00;
			socket_write(client->socket, client->write_buffer, n + 4);
			client->last_write_index = 2;
			client->nwrite = 0;
			client->reset = false;
			return;
		}
		client->last_write_index = 2;
		bolt_reply_structure(client, BST_IGNORED, 0);
		uint16_t n = client->last_write_index - 2;
		*(u_int16_t *)client->write_buffer = htons(n);
		client->write_buffer[n + 2] = 0x00;
		client->write_buffer[n + 3] = 0x00;
		socket_write(client->socket, client->write_buffer, n + 4);
		client->last_write_index = 2;
		bolt_reply_structure(client, BST_SUCCESS, 1);
		bolt_reply_map(client, 0);
		n = client->last_write_index - 2;
		*(u_int16_t *)client->write_buffer = htons(n);
		client->write_buffer[n + 2] = 0x00;
		client->write_buffer[n + 3] = 0x00;
		socket_write(client->socket, client->write_buffer, n + 4);
		client->last_write_index = 2;
		client->nwrite = 0;
		client->reset = false;
		client->state = BS_READY;
		return;
	}

	ASSERT(client->nwrite > 0);
	socket_write(client->socket, client->write_buffer, client->nwrite);
	ASSERT(client->nwrite == client->last_write_index - 2);
	client->nwrite = 0;
	client->last_write_index = 2;
}

bool bolt_check_handshake
(
	socket_t socket
) {
	char data[4];
	int nread = 0;
	while (nread < 4) {
		int n = socket_read(socket, data, 4 - nread);
		if(n > 0) {
			nread += n;
		}
	}
	
	return  data[0] == 0x60 && data[1] == 0x60 && data[2] == (char)0xB0 && data[3] == 0x17;
}

bolt_version_t bolt_read_supported_version
(
	socket_t socket
) {
	char data[16];
	bool res = socket_read(socket, data, 16);
	ASSERT(res);
	bolt_version_t version;
	version.minor = data[2];
	version.major = data[3];
	return version;
}
