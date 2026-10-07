# Locate fields inside a graph's serialized value (encoding v20), so a test can
# corrupt exactly one field of a healthy payload and check the decoder rejects
# it. Works on a DUMP payload and on a key's value inside an RDB file: both hold
# the same bytes.
#
# A graph value is a module type id, a run of RDB strings (one per serializer
# buffer of up to 256000 bytes) and an EOF opcode. Inside the strings every
# value starts with a 1-byte type tag; byte buffers add an 8-byte length.

import struct

# serializer value tags (src/serializers/serializer_io.c)
TAG_BYTES, TAG_FLOAT, TAG_DOUBLE, TAG_SIGNED, TAG_UNSIGNED, TAG_LONG_DOUBLE, \
    TAG_BLOB = range(7)
_SIZES = {TAG_FLOAT: 4, TAG_DOUBLE: 8, TAG_SIGNED: 8, TAG_UNSIGNED: 8,
          TAG_LONG_DOUBLE: 16}

# value types (src/value.h)
T_ARRAY = 1 << 3
T_DATETIME = 1 << 5
T_DATE = 1 << 7
T_TIME = 1 << 8
T_DURATION = 1 << 10
T_STRING = 1 << 11
T_BOOL = 1 << 12
T_INT64 = 1 << 13
T_DOUBLE = 1 << 14
T_NULL = 1 << 15
T_POINT = 1 << 17
T_VECTOR_F32 = 1 << 18
T_INTERN_STRING = (1 << 19) | T_STRING

# payload types (src/serializers/encode_context.h)
NODES, DELETED_NODES, EDGES, DELETED_EDGES = 1, 2, 3, 4
LABELS_MATRICES, RELATION_MATRICES, ADJ_MATRIX, LBLS_MATRIX, CCH_INDICES = \
    6, 7, 8, 9, 10

INDEX_FLD_VECTOR = 0x10

# Redis module type ids: 9 characters of 6 bits each, then a 10-bit encver
_MODULE_CHARSET = ("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
                   "0123456789-_")


def rdb_len(b, p):
    """decode an RDB length at p, returns (length, next offset)"""
    t = b[p]
    if t >> 6 == 0:
        return t & 0x3F, p + 1
    if t >> 6 == 1:
        return ((t & 0x3F) << 8) | b[p + 1], p + 2
    if t == 0x80:
        return int.from_bytes(b[p + 1:p + 5], 'big'), p + 5
    if t == 0x81:
        return int.from_bytes(b[p + 1:p + 9], 'big'), p + 9
    raise ValueError(f'unsupported RDB length encoding {t:#x} at {p} '
                     '(compressed? save with rdbcompression no)')


def rdb_len_bytes(n):
    if n < 1 << 6:
        return bytes([n])
    if n < 1 << 14:
        return bytes([0x40 | (n >> 8), n & 0xFF])
    if n < 1 << 32:
        return b'\x80' + n.to_bytes(4, 'big')
    return b'\x81' + n.to_bytes(8, 'big')


def module_type_name(module_id):
    return ''.join(_MODULE_CHARSET[(module_id >> (64 - 6 * (i + 1))) & 63]
                   for i in range(9))


def module_type_id(name, encver):
    mid = 0
    for ch in name:
        mid = (mid << 6) | _MODULE_CHARSET.index(ch)
    return (mid << 10) | encver


_CRC64_TABLE = None


def crc64(data):
    """CRC-64 (Jones), the checksum Redis stamps into DUMP payloads"""
    global _CRC64_TABLE
    POLY = 0x95ac9329ac4bc9b5
    if _CRC64_TABLE is None:
        table = []
        for i in range(256):
            crc = i
            for _ in range(8):
                crc = (crc >> 1) ^ POLY if (crc & 1) else (crc >> 1)
            table.append(crc)
        _CRC64_TABLE = table
    crc = 0
    for byte in data:
        crc = _CRC64_TABLE[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    return crc & 0xFFFFFFFFFFFFFFFF


def reframe(body, version_bytes):
    """DUMP payload from a module body: <body><2-byte version><8-byte CRC64>"""
    payload = bytes(body) + version_bytes
    return payload + crc64(payload).to_bytes(8, 'little')


def _rdb_string(b, p):
    """read an RDB string at p, any encoding; returns (bytes, next offset)"""
    t = b[p]
    if t >> 6 != 3:
        n, p = rdb_len(b, p)
        return bytes(b[p:p + n]), p + n
    enc = t & 0x3F
    if enc in (0, 1, 2):                     # 8/16/32-bit integer
        n = 1 << enc
        return str(int.from_bytes(b[p + 1:p + 1 + n], 'little',
                                  signed=True)).encode(), p + 1 + n
    if enc == 3:                             # LZF, kept compressed
        clen, q = rdb_len(b, p + 1)
        _, q = rdb_len(b, q)
        return bytes(b[q:q + clen]), q + clen
    raise ValueError(f'unsupported RDB string encoding {t:#x} at {p}')


def _module_value_end(b, p):
    """skip a module value of any shape (p at the module id)"""
    _, p = rdb_len(b, p)
    while True:
        op, p = rdb_len(b, p)
        if op == 0:                          # EOF
            return p
        if op in (1, 2):                     # signed / unsigned
            _, p = rdb_len(b, p)
        elif op == 3:                        # float
            p += 4
        elif op == 4:                        # double
            p += 8
        elif op == 5:                        # string
            n, p = rdb_len(b, p)
            p += n
        else:
            raise ValueError(f'unexpected module opcode {op} at {p}')


def rdb_module_values(b):
    """(key, module type name, value start, value end) for every module key in
    an RDB file; the value starts at the module type id. the file must hold
    only module keys (no streams, hashes, ...)"""
    assert b[:5] == b'REDIS', b[:9]
    p = 9
    out = []
    while True:
        op = b[p]
        p += 1
        if op == 0xFF:                       # EOF
            return out
        if op == 0xFA:                       # aux field
            for _ in range(2):
                _, p = _rdb_string(b, p)
        elif op == 0xF7:                     # module aux
            p = _module_value_end(b, p)
        elif op == 0xFE:                     # select db
            _, p = rdb_len(b, p)
        elif op == 0xFB:                     # resize db
            _, p = rdb_len(b, p)
            _, p = rdb_len(b, p)
        elif op == 0x07:                     # module value
            key, p = _rdb_string(b, p)
            mid, _ = rdb_len(b, p)
            end = _module_value_end(b, p)
            out.append((key, module_type_name(mid), p, end))
            p = end
        else:
            raise ValueError(f'unsupported RDB opcode {op:#x} at {p - 1}')


class Field:
    def __init__(self, tag, off, size, value):
        self.tag = tag      # serializer tag
        self.off = off      # offset of the value's bytes (after tag/length)
        self.size = size    # number of value bytes
        self.value = value  # decoded value


class GraphValue:
    """a graph module value inside 'buf', starting at the module type id"""

    def __init__(self, buf, start, parse=True):
        self.buf = bytearray(buf)
        self.start = start
        _, p = rdb_len(self.buf, start)
        self.chunks = []            # (length prefix offset, data start, end)
        while True:
            op, p = rdb_len(self.buf, p)
            if op == 0:             # RDB_MODULE_OPCODE_EOF
                break
            if op != 5:             # RDB_MODULE_OPCODE_STRING
                raise ValueError(f'unexpected module opcode {op} at {p}')
            q = p
            n, p = rdb_len(self.buf, p)
            self.chunks.append((q, p, p + n))
            p += n
        self.end = p
        self.fields = {}
        if parse:
            self._parse()

    # -- reading ---------------------------------------------------------------

    def _take(self, tag):
        ci, p = self._ci, self._p
        _, d, e = self.chunks[ci]
        if p == e:
            ci += 1
            _, d, e = self.chunks[ci]
            p = d
        t = self.buf[p]
        if t == TAG_BLOB:
            # a blob marker closes a buffer: the next RDB string is the value
            assert tag == TAG_BYTES and p + 1 == e, (tag, p, e)
            ci += 1
            _, d, e = self.chunks[ci]
            f = Field(TAG_BYTES, d, e - d, bytes(self.buf[d:e]))
            self._ci, self._p = ci, e
            return f
        if t != tag:
            raise ValueError(f'expected tag {tag}, found {t} at {p}')
        if t == TAG_BYTES:
            n = int.from_bytes(self.buf[p + 1:p + 9], 'little')
            f = Field(t, p + 9, n, bytes(self.buf[p + 9:p + 9 + n]))
            p += 9 + n
        else:
            n = _SIZES[t]
            raw = bytes(self.buf[p + 1:p + 1 + n])
            if t == TAG_DOUBLE:
                val = struct.unpack('<d', raw)[0]
            elif t == TAG_FLOAT:
                val = struct.unpack('<f', raw)[0]
            else:
                val = int.from_bytes(raw, 'little', signed=(t == TAG_SIGNED))
            f = Field(t, p + 1, n, val)
            p += 1 + n
        self._ci, self._p = ci, p
        return f

    def _u(self, name):
        self.fields[name] = f = self._take(TAG_UNSIGNED)
        return f.value

    def _s(self, name):
        self.fields[name] = f = self._take(TAG_SIGNED)
        return f.value

    def _d(self, name):
        self.fields[name] = f = self._take(TAG_DOUBLE)
        return f.value

    def _b(self, name):
        self.fields[name] = f = self._take(TAG_BYTES)
        return f.value

    # -- layout, in the decoder's order (decoders/current/v20) ------------------

    def _parse(self):
        self._ci, self._p = 0, self.chunks[0][1]
        self._counts = {'node': 0, 'edge': 0, 'tensor': 0}

        # header
        self._b('graph_name')
        for k in ('node_count', 'edge_count', 'deleted_node_count',
                  'deleted_edge_count', 'label_count'):
            self._u(k)
        for r in range(self._u('relation_count')):
            self._u(f'multi_edge[{r}]')
        self._u('key_count')

        # schema
        for a in range(self._u('attribute_count')):
            self._b(f'attribute[{a}]')
        for kind in ('node_schema', 'edge_schema'):
            for k in range(self._u(f'{kind}_count')):
                self._schema(f'{kind}[{k}]')

        # key schema, then the payloads it lists
        payloads = []
        for i in range(self._u('payload_count')):
            payloads.append((self._u(f'payload[{i}].type'),
                             self._u(f'payload[{i}].count')))
        for ptype, count in payloads:
            self._payload(ptype, count)

        # the walk must account for every byte of the value
        last = len(self.chunks) - 1
        if (self._ci, self._p) != (last, self.chunks[last][2]):
            raise ValueError(f'walk stopped at {self._p}, value continues '
                             f'(chunk {self._ci} of {len(self.chunks)})')

    def _schema(self, n):
        self._u(f'{n}.id')
        self._b(f'{n}.name')
        for x in range(self._u(f'{n}.index_count')):
            ix = f'{n}.index' if x == 0 else f'{n}.index{x}'
            self._b(f'{ix}.language')
            for s in range(self._u(f'{ix}.stopword_count')):
                self._b(f'{ix}.stopword[{s}]')
            for i in range(self._u(f'{ix}.field_count')):
                fn = f'{ix}.field[{i}]'
                self._b(f'{fn}.name')
                ftype = self._u(f'{fn}.type')
                self._d(f'{fn}.weight')
                self._u(f'{fn}.nostem')
                self._b(f'{fn}.phonetic')
                if ftype & INDEX_FLD_VECTOR:
                    for o in ('dimension', 'M', 'ef_construction',
                              'ef_runtime', 'similarity'):
                        self._u(f'{fn}.{o}')
        for c in range(self._u(f'{n}.constraint_count')):
            cn = f'{n}.constraint[{c}]'
            self._u(f'{cn}.type')
            for a in range(self._u(f'{cn}.attribute_count')):
                self._u(f'{cn}.attribute[{a}]')

    def _payload(self, ptype, count):
        if ptype == NODES:
            self._entities('node', count)
        elif ptype == EDGES:
            self._entities('edge', count)
        elif ptype == DELETED_NODES:
            self._b('deleted_nodes')
        elif ptype == DELETED_EDGES:
            self._b('deleted_edges')
        elif ptype == LABELS_MATRICES:
            for k in range(self._u('label_matrix_count')):
                self._u(f'label_matrix[{k}].label')
                self._delta(f'label_matrix[{k}]')
        elif ptype == RELATION_MATRICES:
            for r in range(self.fields['edge_schema_count'].value):
                rn = f'relation_matrix[{r}]'
                self._u(f'{rn}.relation')
                self._delta(rn)
                if self._u(f'{rn}.tensor_count'):
                    for part in ('M', 'DP'):
                        for _ in range(self._u(f'{rn}.{part}.tensor_count')):
                            tn = f'tensor[{self._counts["tensor"]}]'
                            self._counts['tensor'] += 1
                            self._u(f'{tn}.i')
                            self._u(f'{tn}.j')
                            self._b(f'{tn}.blob')
        elif ptype == ADJ_MATRIX:
            self._delta('adjacency')
        elif ptype == LBLS_MATRIX:
            self._delta('labels')
        elif ptype == CCH_INDICES:
            for k in range(self._u('cch_count')):
                for r in range(self._u(f'cch[{k}].relation_count')):
                    self._u(f'cch[{k}].relation[{r}]')
                self._u(f'cch[{k}].weight_attribute')
        else:
            raise ValueError(f'unknown payload type {ptype}')

    def _entities(self, kind, count):
        for _ in range(count):
            en = f'{kind}[{self._counts[kind]}]'
            self._counts[kind] += 1
            self._u(f'{en}.id')
            for p in range(self._u(f'{en}.property_count')):
                self._u(f'{en}.property[{p}].attribute')
                self._value(f'{en}.property[{p}]')

    def _value(self, n):
        t = self._u(f'{n}.type')
        if t in (T_INT64, T_BOOL, T_TIME, T_DATE, T_DATETIME, T_DURATION):
            self._s(f'{n}.value')
        elif t == T_DOUBLE:
            self._d(f'{n}.value')
        elif t in (T_STRING, T_INTERN_STRING, T_VECTOR_F32):
            self._b(f'{n}.value')
        elif t == T_POINT:
            self._d(f'{n}.lat')
            self._d(f'{n}.lon')
        elif t == T_ARRAY:
            for e in range(self._u(f'{n}.length')):
                self._value(f'{n}.element[{e}]')
        elif t != T_NULL:
            raise ValueError(f'unknown value type {t} at {n}')

    def _delta(self, n):
        for m in ('M', 'DP', 'DM'):
            mn = f'{n}.{m}'
            self._b(f'{mn}.container')
            for v in ('x', 'h', 'p', 'i', 'b'):
                vn = f'{mn}.{v}'
                self._b(f'{vn}.data')
                self._b(f'{vn}.type')
                self._u(f'{vn}.entries')
                self._u(f'{vn}.bytes')
                self._s(f'{vn}.handling')

    # -- edits -----------------------------------------------------------------

    def set_uint(self, name, value):
        f = self.fields[name]
        assert f.tag == TAG_UNSIGNED, name
        self.buf[f.off:f.off + 8] = value.to_bytes(8, 'little')

    def set_tag(self, name, tag):
        """overwrite a value's serializer type tag"""
        f = self.fields[name]
        self.buf[f.off - (9 if f.tag == TAG_BYTES else 1)] = tag

    def set_length(self, name, length):
        """overwrite a byte buffer's declared length, keeping its bytes"""
        f = self.fields[name]
        assert f.tag == TAG_BYTES, name
        self.buf[f.off - 8:f.off] = length.to_bytes(8, 'little')

    def set_bytes(self, offset, data):
        self.buf[offset:offset + len(data)] = data

    def remove_value(self, name):
        """drop a value (tag and bytes), fixing the RDB string length around
        it; field offsets after it are stale afterwards"""
        self.remove_values(name, name)

    def remove_values(self, first, last):
        """drop the values from 'first' through 'last', both included"""
        f, l = self.fields[first], self.fields[last]
        start = f.off - (9 if f.tag == TAG_BYTES else 1)
        end = l.off + l.size
        for q, d, e in self.chunks:
            if d <= start and end <= e:
                prefix = rdb_len_bytes((e - d) - (end - start))
                self.buf[start:end] = b''
                self.buf[q:d] = prefix
                return
        raise ValueError(f'{first}..{last} spans RDB strings')

    def find(self, suffix, value):
        """names of the fields ending with 'suffix' whose value is 'value'"""
        return [n for n, f in self.fields.items()
                if n.endswith(suffix) and f.value == value]
