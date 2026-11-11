# Moving a data volume from Redis to Valkey

FalkorDB images ship one server, Valkey or Redis. Valkey cannot load the save
files that Redis 7.4 and later write (RDB version 12 and up), and the
FalkorDB images until now ran Redis 8. So a volume written by one of those
images needs one migration before a Valkey image can use it.

To check which server an image runs (`valkey` or `redis`; images built before
the label existed print `<no value>` and run Redis):

```sh
docker pull falkordb/falkordb-server:<tag>
docker image inspect -f '{{index .Config.Labels "com.falkordb.server.name"}}' falkordb/falkordb-server:<tag>
```

## What a Valkey image does with Redis data

Before Valkey starts, the image reads the header of every file in the data
directory and of the `*.base.rdb` files in the directories under it (an AOF's
base). If Redis wrote one, the container stops before Valkey starts, prints
the file names and the link to this page, and leaves the volume as it was. An
AOF whose base is commands rather than an RDB is replayed by Valkey as it is.

To keep the data as it is, run a Redis image of FalkorDB on the volume (one
whose `com.falkordb.server.name` label is `redis`). Otherwise migrate it:

## Migrate to Valkey

The commands use `docker exec` on a container named `falkordb`; use your own
name. Wherever `docker run` says "your other options", repeat those of your
Redis container (`REDIS_ARGS`/`VALKEY_ARGS`, `TLS`, password or ACL settings,
mounts, ports), minus any option only Redis has: Valkey refuses to start on
one it does not know. The data must be in a named volume or a bind mount.
With a password, pass `-e REDISCLI_AUTH=<password>` to each `docker exec`
(and `--user <user>` to `redis-cli` for an ACL user). With `TLS=1`, run
`redis-cli` through `sh -c 'redis-cli --tls --cacert "$FALKORDB_TLS_PATH/ca.crt" ...'`.

1. **Back up the volume**, and stop your applications from writing until
   step 3 has finished: later writes are not in the files Valkey loads.

2. **On the Redis image**, remove what Valkey cannot read, and save. FalkorDB
   keeps each graph's query statistics in streams named `telemetry{<graph>}`,
   which Valkey cannot read. Stop new ones, then delete them (only streams: a
   graph or string with such a name is yours):

   ```sh
   docker exec falkordb redis-cli GRAPH.CONFIG SET CMD_INFO no
   docker exec falkordb sh -c \
     'redis-cli --scan --pattern "telemetry{*}" | while IFS= read -r key; do [ "$(redis-cli TYPE "$key")" = stream ] && redis-cli DEL "$key"; done'
   ```

   This covers database 0, where FalkorDB keeps its keys; repeat with
   `redis-cli -n <db>` for any other database you use. If you keep other data
   next to your graphs, also delete or convert other streams, hashes with
   field expiry, and keys of a type only Redis has, such as arrays. Then save,
   and with AOF on, rewrite the AOF and wait until `aof_rewrite_in_progress`
   and `aof_rewrite_scheduled` are `0` and `aof_last_bgrewrite_status` is `ok`:

   ```sh
   docker exec falkordb redis-cli SAVE
   docker exec falkordb redis-cli BGREWRITEAOF   # AOF only
   docker exec falkordb redis-cli INFO persistence | grep -E 'aof_rewrite_(scheduled|in_progress)|aof_last_bgrewrite_status'
   docker stop falkordb && docker rm falkordb
   ```

3. **Start the Valkey image with `FALKORDB_REDIS_MIGRATION=1`** on the same
   volume, which starts Valkey with `--rdb-version-check relaxed`:

   ```sh
   docker run -d --name falkordb -v <volume>:/var/lib/falkordb/data \
     -e FALKORDB_REDIS_MIGRATION=1 <your other options> \
     falkordb/falkordb-server:<valkey tag>
   ```

   Check that your graphs are there (`GRAPH.LIST`, and a query), then save
   them in Valkey's format with `SAVE` (and `BGREWRITEAOF` with AOF on, as in
   step 2). If Valkey stops with `Can't handle foreign type or opcode`, a key
   it cannot read is still in the files: go back to step 2.

4. **Start the Valkey image again without the variable.** `docker restart`
   would keep it, so remove the container and run it again without
   `-e FALKORDB_REDIS_MIGRATION=1`:

   ```sh
   docker stop falkordb && docker rm falkordb
   docker run -d --name falkordb -v <volume>:/var/lib/falkordb/data \
     <your other options> falkordb/falkordb-server:<valkey tag>
   ```

Query statistics start again from empty. Redis images ignore
`FALKORDB_REDIS_MIGRATION`.
