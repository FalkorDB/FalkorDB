"""`BenchClient.raw` turns a nil reply into an error.

DUMP of an absent key answers nil, not an error reply, so without this a
`--reuse` run against a server that lacks the benchmark graph would record a
serialization row (and `--once` a clean validation) for a command that
serialized nothing.
"""

import pytest
from redis.exceptions import ResponseError

from falkorbench.client import BenchClient


class _Db:
    def __init__(self, reply):
        self.reply = reply

    def execute_command(self, *args, **kwargs):
        return self.reply


def _client(reply):
    client = object.__new__(BenchClient)
    client.db = _Db(reply)
    return client


def test_nil_reply_is_an_error():
    with pytest.raises(ResponseError):
        _client(None).raw("DUMP", "g")


def test_binary_reply_passes_through():
    assert _client(b"\x00\x01").raw("DUMP", "g") == b"\x00\x01"
