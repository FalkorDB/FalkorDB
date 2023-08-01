#include "RG.h"
#include "bolt.h"
#include "string.h"
#include "util/rmalloc.h"
#include <byteswap.h>

bolt_client_t *bolt_client_new
(
    socket_t socket,
    RedisModuleEventLoopFunc on_write
) {
    bolt_client_t *client = rm_malloc(sizeof(bolt_client_t));
    client->socket = socket;
    client->state = BS_NEGOTIATION;
    client->on_write = on_write;
    client->write_index = 2;
    client->read_index = 0;
    return client;
}

void bolt_change_negotiation_state
(
    bolt_client_t *client   
) {
    ASSERT(client->state == BS_NEGOTIATION && bolt_value_get_structure_type(client->read_buffer) == BST_HELLO);
    bolt_structure_type response_type = bolt_value_get_structure_type(client->write_buffer + 2);
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
    bolt_client_t *client   
) {
    ASSERT(client->state == BS_AUTHENTICATION && bolt_value_get_structure_type(client->read_buffer) == BST_LOGON);
    bolt_structure_type response_type = bolt_value_get_structure_type(client->write_buffer + 2);
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
    bolt_client_t *client   
) {
    ASSERT(client->state == BS_READY);
    bolt_structure_type request_type = bolt_value_get_structure_type(client->read_buffer);
    bolt_structure_type response_type = bolt_value_get_structure_type(client->write_buffer + 2);
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
    bolt_client_t *client   
) {
    ASSERT(client->state == BS_STREAMING);
    bolt_structure_type request_type = bolt_value_get_structure_type(client->read_buffer);
    bolt_structure_type response_type = bolt_value_get_structure_type(client->write_buffer + 2);
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
    bolt_client_t *client   
) {
    ASSERT(client->state == BS_TX_READY);
    bolt_structure_type request_type = bolt_value_get_structure_type(client->read_buffer);
    bolt_structure_type response_type = bolt_value_get_structure_type(client->write_buffer + 2);
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
    bolt_client_t *client   
) {
    ASSERT(client->state == BS_TX_STREAMING);
    bolt_structure_type request_type = bolt_value_get_structure_type(client->read_buffer);
    bolt_structure_type response_type = bolt_value_get_structure_type(client->write_buffer + 2);
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
    bolt_client_t *client   
) {
    ASSERT(client->state == BS_FAILED);
    bolt_structure_type request_type = bolt_value_get_structure_type(client->read_buffer);
    bolt_structure_type response_type = bolt_value_get_structure_type(client->write_buffer + 2);
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
    bolt_client_t *client   
) {
    ASSERT(client->state == BS_INTERRUPTED);
    bolt_structure_type request_type = bolt_value_get_structure_type(client->read_buffer);
    bolt_structure_type response_type = bolt_value_get_structure_type(client->write_buffer + 2);
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
    bolt_client_t *client   
) {
    bolt_structure_type response_type = bolt_value_get_structure_type(client->write_buffer + 2);
    if(response_type == BST_RECORD) {
        return;
    }
    switch (client->state)
	{
        case BS_NEGOTIATION:
            bolt_change_negotiation_state(client);
            break;
        case BS_AUTHENTICATION:
            bolt_change_authentication_state(client);
            break;
        case BS_READY:
            bolt_change_ready_state(client);
            break;
        case BS_STREAMING:
            bolt_change_streaming_state(client);
            break;
        case BS_TX_READY:
            bolt_change_txready_state(client);
            break;
        case BS_TX_STREAMING:
            bolt_change_txstreaming_state(client);
            break;
        case BS_FAILED:
            bolt_change_failed_state(client);
            break;
        case BS_INTERRUPTED:
            bolt_change_interrupted_state(client);
            break;
        default:
            ASSERT(false);
            break;
    }
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
    if(client->state == BS_FAILED && bolt_value_get_structure_type(client->read_buffer) != BST_RESET) {
        client->write_index = 2;
		bolt_reply_structure(client, BST_IGNORED, 0);
	}

    uint16_t n = client->write_index - 2;
    if(n == 0) {
        return;
    }
    *(u_int16_t *)client->write_buffer = bswap_16(n);
    client->write_buffer[n + 2] = 0x00;
	client->write_buffer[n + 3] = 0x00;
    socket_write(client->socket, client->write_buffer, n + 4);
    client->write_index = 2;
    bolt_change_client_state(client);
}

void bolt_reply_null
(
    bolt_client_t *client
) {
    client->write_buffer[client->write_index++] = 0xC0;
}

void bolt_reply_bool
(
    bolt_client_t *client,
    bool data
) {
    client->write_buffer[client->write_index++] = data ? 0xC3 : 0xC2;
}

void bolt_reply_int8
(
    bolt_client_t *client,
    int8_t data
) {
    client->write_buffer[client->write_index++] = 0xC8;
    client->write_buffer[client->write_index++] = data;
}

void bolt_reply_int16
(
    bolt_client_t *client,
    int16_t data
) {
    client->write_buffer[client->write_index++] = 0xC9;
    *(uint16_t *)(client->write_buffer + client->write_index) = data;
    client->write_index += 2;
}

void bolt_reply_int32
(
    bolt_client_t *client,
    int32_t data
) {
    client->write_buffer[client->write_index++] = 0xCA;
    *(uint32_t *)(client->write_buffer + client->write_index) = data;
    client->write_index += 4;
}

void bolt_reply_int64
(
    bolt_client_t *client,
    int64_t data
) {
    client->write_buffer[client->write_index++] = 0xCB;
    *(uint64_t *)(client->write_buffer + client->write_index) = data;
    client->write_index += 8;
}

void bolt_reply_float
(
    bolt_client_t *client,
    double data
) {
    client->write_buffer[client->write_index++] = 0xC1;
    char *buf = (char *)&data;
    for (int i = 0; i < sizeof(double); i++)
      client->write_buffer[client->write_index++] = buf[sizeof(double) - i - 1];
}

void bolt_reply_string
(
    bolt_client_t *client,
    const char *data
) {
    uint32_t size = strlen(data);
    if (size < 0x10) {
        client->write_buffer[client->write_index++] = 0x80 + size;
        memcpy(client->write_buffer + client->write_index, data, size);
        client->write_index += size;
    } else if (size < 0x100) {
        client->write_buffer[client->write_index++] = 0xD0;
        client->write_buffer[client->write_index++] = size;
        memcpy(client->write_buffer + client->write_index, data, size);
        client->write_index += size;
    } else if (size < 0x10000) {
        client->write_buffer[client->write_index++] = 0xD1;
        *(uint16_t *)(client->write_buffer + client->write_index) = size;
        client->write_index += 2;
        memcpy(client->write_buffer + client->write_index, data, size);
        client->write_index += size;
    } else {
        client->write_buffer[client->write_index++] = 0xD2;
        *(uint32_t *)(client->write_buffer + client->write_index) = size;
        client->write_index += 4;
        memcpy(client->write_buffer + client->write_index, data, size);
        client->write_index += size;
    }
}

void bolt_reply_list
(
    bolt_client_t *client,
    uint32_t size
) {
    if (size < 0x10) {
        client->write_buffer[client->write_index++] = 0x90 + size;
    } else if (size < 0x100) {
        client->write_buffer[client->write_index++] = 0xD4;
        client->write_buffer[client->write_index++] = size;
    } else if (size < 0x10000) {
        client->write_buffer[client->write_index++] = 0xD5;
        *(uint16_t *)(client->write_buffer + client->write_index) = size;
        client->write_index += 2;
    } else {
        client->write_buffer[client->write_index++] = 0xD6;
        *(uint32_t *)(client->write_buffer + client->write_index) = size;
        client->write_index += 4;
    }
}

void bolt_reply_map
(
    bolt_client_t *client,
    uint32_t size
) {
    if (size < 0x10) {
        client->write_buffer[client->write_index++] = 0xA0 + size;
    } else if (size < 0x100) {
        client->write_buffer[client->write_index++] = 0xD8;
        client->write_buffer[client->write_index++] = size;
    } else if (size < 0x10000) {
        client->write_buffer[client->write_index++] = 0xD9;
        *(uint16_t *)(client->write_buffer + client->write_index) = size;
        client->write_index += 2;
    } else {
        client->write_buffer[client->write_index++] = 0xDA;
        *(uint32_t *)(client->write_buffer + client->write_index) = size;
        client->write_index += 4;
    }
}

void bolt_reply_structure
(
    bolt_client_t *client,
    bolt_structure_type type,
    uint32_t size
) {
    client->write_buffer[client->write_index++] = 0xB0 + size;
    client->write_buffer[client->write_index++] = type;
}

char *bolt_value_read
(
    char *data
) {
    uint8_t marker = data[0];
    switch (marker)
    {
        case 0x80:
        case 0x81:
        case 0x82:
        case 0x83:
        case 0x84:
        case 0x85:
        case 0x86:
        case 0x87:
        case 0x88:
        case 0x89:
        case 0x8A:
        case 0x8B:
        case 0x8C:
        case 0x8D:
        case 0x8E:
        case 0x8F:
            return data + 1 + marker - 0x80;
        case 0x90:
        case 0x91:
        case 0x92:
        case 0x93:
        case 0x94:
        case 0x95:
        case 0x96:
        case 0x97:
        case 0x98:
        case 0x99:
        case 0x9A:
        case 0x9B:
        case 0x9C:
        case 0x9D:
        case 0x9E:
        case 0x9F:
            data = data + 1;
            for (uint32_t i = 0; i < marker - 0x90; i++) {
                data = bolt_value_read(data);
            }
            return data;
        case 0xA0:
        case 0xA1:
        case 0xA2:
        case 0xA3:
        case 0xA4:
        case 0xA5:
        case 0xA6:
        case 0xA7:
        case 0xA8:
        case 0xA9:
        case 0xAA:
        case 0xAB:
        case 0xAC:
        case 0xAD:
        case 0xAE:
        case 0xAF:
            data = data + 1;
            for (uint32_t i = 0; i < marker - 0xA0; i++) {
                data = bolt_value_read(data);
                data = bolt_value_read(data);
            }
            return data;
        case 0xB0:
        case 0xB1:
        case 0xB2:
        case 0xB3:
        case 0xB4:
        case 0xB5:
        case 0xB6:
        case 0xB7:
        case 0xB8:
        case 0xB9:
        case 0xBA:
        case 0xBB:
        case 0xBC:
        case 0xBD:
        case 0xBE:
        case 0xBF:
            data = data + 2;
            for (uint32_t i = 0; i < marker - 0xB0; i++) {
                data = bolt_value_read(data);
            }
            return data;
        case 0xC0:
            return data + 1;
        case 0xC1:
            return data + 9;
        case 0xC2:
            return data + 1;
        case 0xC3:
            return data + 1;
        case 0xC8:
            return data + 2;
        case 0xC9:
            return data + 3;
        case 0xCA:
            return data + 5;
        case 0xCB:
            return data + 9;
        case 0xCC:
            return data + 2 + *(uint8_t *)(data + 1);
        case 0xCD:
            return data + 3 + *(uint16_t *)(data + 1);
        case 0xCE:
            return data + 5 + *(uint32_t *)(data + 1);
        case 0xD0:
            return data + 2 + *(uint8_t *)(data + 1);
        case 0xD1:
            return data + 3 + bswap_16(*(uint16_t *)(data + 1));
        case 0xD2:
            return data + 5 + *(uint32_t *)(data + 1);
        case 0xD4: {
            int n = data[1];
            data = data + 2;
            for (uint32_t i = 0; i < n; i++) {
                data = bolt_value_read(data);
            }
            return data;
        }
        case 0xD5: {
            int n = *(uint16_t *)(data + 1);
            data = data + 3;
            for (uint32_t i = 0; i < n; i++) {
                data = bolt_value_read(data);
            }
            return data;
        }
        case 0xD6: {
            int n = *(uint32_t *)(data + 1);
            data = data + 5;
            for (uint32_t i = 0; i < n; i++) {
                data = bolt_value_read(data);
            }
            return data;
        }
        default:
            if(marker >= 0xF0 || marker <= 0x7F) {
                return data + 1;
            }
            ASSERT(false);
            break;
    }
}

bolt_value_type bolt_value_get_type
(
    char *data
) {
    uint8_t marker = data[0];
    switch (marker)
    {
        case 0xC0:
            return BVT_NULL;
        case 0xC1:
            return BVT_FLOAT;
        case 0xC2:
        case 0xC3:
            return BVT_BOOL;
        case 0xC8:
            return BVT_INT8;
        case 0xC9:
            return BVT_INT16;
        case 0xCA:
            return BVT_INT32;
        case 0xCB:
            return BVT_INT64;
        case 0xCC:
        case 0xCD:
        case 0xCE:
            return BVT_BYTES;
        case 0x80:
        case 0x81:
        case 0x82:
        case 0x83:
        case 0x84:
        case 0x85:
        case 0x86:
        case 0x87:
        case 0x88:
        case 0x89:
        case 0x8A:
        case 0x8B:
        case 0x8C:
        case 0x8D:
        case 0x8E:
        case 0x8F:
        case 0xD0:
        case 0xD1:
        case 0xD2:
            return BVT_STRING;
        case 0xD4:
        case 0xD5:
        case 0xD6:
            return BVT_LIST;
        case 0xD8:
        case 0xD9:
        case 0xDA:
            return BVT_MAP;
        case 0xB0:
        case 0xB1:
        case 0xB2:
        case 0xB3:
        case 0xB4:
        case 0xB5:
        case 0xB6:
        case 0xB7:
        case 0xB8:
        case 0xB9:
        case 0xBA:
        case 0xBB:
        case 0xBC:
        case 0xBD:
        case 0xBE:
        case 0xBF:
            return BVT_STRUCTURE;
        default:
            if(marker >= 0xF0 || marker <= 0x7F) {
                return BVT_INT8;
            }
            break;
    }
}

bool bolt_value_get_bool
(
    char *data
) {
    uint8_t marker = data[0];
    switch (marker)
    {
        case 0xC2:
            return false;
        case 0xC3:
            return true;
        default:
            ASSERT(false);
            return false;
    }
}

int8_t bolt_value_get_int8
(
    char *data
) {
    uint8_t marker = data[0];
    switch (marker)
    {
        case 0xC8:
            return data[1];
        default:
            ASSERT(false);
            return 0;
    }
}

int16_t bolt_value_get_int16
(
    char *data
) {
    uint8_t marker = data[0];
    switch (marker)
    {
        case 0xC9:
            return *(uint16_t *)(data + 1);
        default:
            ASSERT(false);
            return 0;
    }
}

int32_t bolt_value_get_int32
(
    char *data
) {
    uint8_t marker = data[0];
    switch (marker)
    {
        case 0xCA:
            return *(uint32_t *)(data + 1);
        default:
            ASSERT(false);
            return 0;
    }
}

int64_t bolt_value_get_int64
(
    char *data
) {
    uint8_t marker = data[0];
    switch (marker)
    {
        case 0xCB:
            return *(uint64_t *)(data + 1);
        default:
            ASSERT(false);
            return 0;
    }
}

double bolt_value_get_float
(
    char *data
) {
    uint8_t marker = data[0];
    switch (marker)
    {
        case 0xC1:
            return *(double *)(data + 1);
        default:
            ASSERT(false);
            return 0;
    }
}

uint32_t bolt_value_get_string_size
(
    char *data
) {
    uint8_t marker = data[0];
    switch (marker)
    {
        case 0x80:
        case 0x81:
        case 0x82:
        case 0x83:
        case 0x84:
        case 0x85:
        case 0x86:
        case 0x87:
        case 0x88:
        case 0x89:
        case 0x8A:
        case 0x8B:
        case 0x8C:
        case 0x8D:
        case 0x8E:
        case 0x8F:
            return marker - 0x80;
        case 0xD0:
            return *(uint8_t *)(data + 1);
        case 0xD1:
            return *(uint16_t *)(data + 1);
        case 0xD3:
            return *(uint32_t *)(data + 1);
        default:
            ASSERT(false);
            return 0;
    }
}

char *bolt_value_get_string
(
    char *data
) {
    uint8_t marker = data[0];
    switch (marker)
    {
        case 0x80:
        case 0x81:
        case 0x82:
        case 0x83:
        case 0x84:
        case 0x85:
        case 0x86:
        case 0x87:
        case 0x88:
        case 0x89:
        case 0x8A:
        case 0x8B:
        case 0x8C:
        case 0x8D:
        case 0x8E:
        case 0x8F:
            return data + 1;
        case 0xD0:
            return data + 2;
        case 0xD1:
            return data + 3;
        case 0xD3:
            return data + 5;
        default:
            ASSERT(false);
            return 0;
    }
}

uint32_t bolt_value_get_list_size
(
    char *data
) {
    uint8_t marker = data[0];
    switch (marker)
    {
        case 0x90:
        case 0x91:
        case 0x92:
        case 0x93:
        case 0x94:
        case 0x95:
        case 0x96:
        case 0x97:
        case 0x98:
        case 0x99:
        case 0x9A:
        case 0x9B:
        case 0x9C:
        case 0x9D:
        case 0x9E:
        case 0x9F:
            return marker - 0x90;
        case 0xD4:
            return *(uint8_t *)(data + 1);
        case 0xD5:
            return *(uint16_t *)(data + 1);
        case 0xD6:
            return *(uint32_t *)(data + 1);
        default:
            ASSERT(false);
            return 0;
    }
}

char *bolt_value_get_list_item
(
    char *data,
    uint32_t index
) {
    uint8_t marker = data[0];
    switch (marker)
    {
        case 0x90:
        case 0x91:
        case 0x92:
        case 0x93:
        case 0x94:
        case 0x95:
        case 0x96:
        case 0x97:
        case 0x98:
        case 0x99:
        case 0x9A:
        case 0x9B:
        case 0x9C:
        case 0x9D:
        case 0x9E:
        case 0x9F:
            data = data + 1;
            for (uint32_t i = 0; i < index; i++) {
                data = bolt_value_read(data);
            }
            return data;
        case 0xD4:
            data = data + 2;
            for (uint32_t i = 0; i < index; i++) {
                data = bolt_value_read(data);
            }
            return data;
        case 0xD5:
            data = data + 3;
            for (uint32_t i = 0; i < index; i++) {
                data = bolt_value_read(data);
            }
            return data;
        case 0xD6:
            data = data + 5;
            for (uint32_t i = 0; i < index; i++) {
                data = bolt_value_read(data);
            }
            return data;
        default:
            ASSERT(false);
            return 0;
    }
}

uint32_t bolt_value_get_map_size
(
    char *data
) {
    uint8_t marker = data[0];
    switch (marker)
    {
        case 0xA0:
        case 0xA1:
        case 0xA2:
        case 0xA3:
        case 0xA4:
        case 0xA5:
        case 0xA6:
        case 0xA7:
        case 0xA8:
        case 0xA9:
        case 0xAA:
        case 0xAB:
        case 0xAC:
        case 0xAD:
        case 0xAE:
        case 0xAF:
            return marker - 0xA0;
        case 0xD8:
            return *(uint8_t *)(data + 1);
        case 0xD9:
            return *(uint16_t *)(data + 1);
        case 0xDA:
            return *(uint32_t *)(data + 1);
        default:
            ASSERT(false);
            return 0;
    }
}

char *bolt_value_get_map_key
(
    char *data,
    uint32_t index
) {
    uint8_t marker = data[0];
    switch (marker)
    {
        case 0xA0:
        case 0xA1:
        case 0xA2:
        case 0xA3:
        case 0xA4:
        case 0xA5:
        case 0xA6:
        case 0xA7:
        case 0xA8:
        case 0xA9:
        case 0xAA:
        case 0xAB:
        case 0xAC:
        case 0xAD:
        case 0xAE:
        case 0xAF:
            data = data + 1;
            for (uint32_t i = 0; i < index; i++) {
                data = bolt_value_read(data);
                data = bolt_value_read(data);
            }
            return data;
        case 0xD8:
            data = data + 2;
            for (uint32_t i = 0; i < index; i++) {
                data = bolt_value_read(data);
                data = bolt_value_read(data);
            }
            return data;
        case 0xD9:
            data = data + 3;
            for (uint32_t i = 0; i < index; i++) {
                data = bolt_value_read(data);
                data = bolt_value_read(data);
            }
            return data;
        case 0xDA:
            data = data + 5;
            for (uint32_t i = 0; i < index; i++) {
                data = bolt_value_read(data);
                data = bolt_value_read(data);
            }
            return data;
        default:
            ASSERT(false);
            return 0;
    }
}

char *bolt_value_get_map_value
(
    char *data,
    uint32_t index
) {
    uint8_t marker = data[0];
    switch (marker)
    {
        case 0xA0:
        case 0xA1:
        case 0xA2:
        case 0xA3:
        case 0xA4:
        case 0xA5:
        case 0xA6:
        case 0xA7:
        case 0xA8:
        case 0xA9:
        case 0xAA:
        case 0xAB:
        case 0xAC:
        case 0xAD:
        case 0xAE:
        case 0xAF:
            data = data + 1;
            for (uint32_t i = 0; i < index; i++) {
                data = bolt_value_read(data);
                data = bolt_value_read(data);
            }
            data = bolt_value_read(data);
            return data;
        case 0xD8:
            data = data + 2;
            for (uint32_t i = 0; i < index; i++) {
                data = bolt_value_read(data);
                data = bolt_value_read(data);
            }
            data = bolt_value_read(data);
            return data;
        case 0xD9:
            data = data + 3;
            for (uint32_t i = 0; i < index; i++) {
                data = bolt_value_read(data);
                data = bolt_value_read(data);
            }
            data = bolt_value_read(data);
            return data;
        case 0xDA:
            data = data + 5;
            for (uint32_t i = 0; i < index; i++) {
                data = bolt_value_read(data);
                data = bolt_value_read(data);
            }
            data = bolt_value_read(data);
            return data;
        default:
            ASSERT(false);
            return 0;
    }
}

bolt_structure_type bolt_value_get_structure_type
(
    char *data
) {
    uint8_t marker = data[0];
    switch (marker)
    {
        case 0xB0:
        case 0xB1:
        case 0xB2:
        case 0xB3:
        case 0xB4:
        case 0xB5:
        case 0xB6:
        case 0xB7:
        case 0xB8:
            return data[1];
        default:
            ASSERT(false);
            return 0;
    }
}

uint32_t bolt_value_get_structure_size
(
    char *data
) {
    uint8_t marker = data[0];
    switch (marker)
    {
        case 0xB0:
        case 0xB1:
        case 0xB2:
        case 0xB3:
        case 0xB4:
        case 0xB5:
        case 0xB6:
        case 0xB7:
        case 0xB8:
        case 0xB9:
        case 0xBA:
        case 0xBB:
        case 0xBC:
        case 0xBD:
        case 0xBE:
        case 0xBF:
            return marker - 0xB0;
        default:
            ASSERT(false);
            return 0;
    }
}

char *bolt_value_get_structure_value
(
    char *data,
    uint32_t index
) {
    uint8_t marker = data[0];
    switch (marker)
    {
        case 0xB0:
        case 0xB1:
        case 0xB2:
        case 0xB3:
        case 0xB4:
        case 0xB5:
        case 0xB6:
        case 0xB7:
        case 0xB8:
        case 0xB9:
        case 0xBA:
        case 0xBB:
        case 0xBC:
        case 0xBD:
        case 0xBE:
        case 0xBF:
            data = data + 2;
            for (uint32_t i = 0; i < index; i++) {
                data = bolt_value_read(data);
            }
            return data;
        default:
            ASSERT(false);
            return 0;
    }
}

bool bolt_check_handshake
(
    socket_t socket
) {
    char data[4];
    int nread = socket_read(socket, data, 4);
    return nread == 4 && data[0] == 0x60 && data[1] == 0x60 && data[2] == (char)0xB0 && data[3] == 0x17;
}

bolt_version_t bolt_read_supported_version
(
    socket_t socket
) {
    char data[16];
    socket_read(socket, data, 16);
    bolt_version_t version;
    version.minor = data[2];
    version.major = data[3];
    return version;
}
