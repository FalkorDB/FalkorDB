# Moving a data volume from Redis to Valkey

FalkorDB images ship one server, Valkey or Redis. Valkey cannot load the save
files that Redis 7.4 and later write (RDB version 12 and up), and the
FalkorDB images until now ran Redis 8. So a volume written by one of those
images needs one migration before a Valkey image can use it.

To check which server an image runs:

```sh
docker pull falkordb/falkordb-server:<tag>
docker image inspect -f '{{index .Config.Labels "com.falkordb.server.name"}}' falkordb/falkordb-server:<tag>
```

It prints `valkey` or `redis`. Images built before this label existed print
`<no value>`, and all of them run Redis.

## What a Valkey image does with Redis data

Before Valkey starts, the image finds the file Valkey would load:

- with AOF off (the default), the RDB file, `dump.rdb` unless `dbfilename`
  says otherwise;
- with AOF on, the base file the AOF manifest lists (with no manifest, the
  legacy single-file AOF).

It reads the configuration the way Valkey does: the config file given in
`VALKEY_ARGS` (or `REDIS_ARGS`), the files it includes, and the options after
it. If Redis wrote that file, the container stops before Valkey starts,
prints the file and these steps, and leaves the volume as it was.

An AOF whose base is commands rather than an RDB (`aof-use-rdb-preamble no`)
is replayed by Valkey as it is, and needs no migration, as long as it holds
no command that only Redis has.

## Keep running Redis

Use a Redis image of FalkorDB on the volume: one whose
`com.falkordb.server.name` label is `redis` (see above for how to check).
Nothing needs to change.

## Migrate to Valkey

The commands below use `docker exec` on a container named `falkordb`; replace
the name with yours. If you set `REDIS_ARGS`, the Valkey image reads it too
when `VALKEY_ARGS` is not set, so start it with the same setting.

Wherever a `docker run` below says "your other options", repeat the options of
your Redis container: `REDIS_ARGS`/`VALKEY_ARGS`, `TLS` and its certificates,
password or ACL settings and mounts, ports. Only `FALKORDB_REDIS_MIGRATION`
is added in step 3 and left out in step 4. The data must be in a named volume
or a bind mount: a container started with `--rm` and an anonymous volume
loses its data when stopped, so copy the data out first.

1. **Back up the volume.** The steps below rewrite its files. Stop your
   applications from writing to FalkorDB until step 3 has finished: writes
   after the `SAVE` in step 2 are not in the files Valkey loads.

   The `docker exec` examples below assume the default plain connection. With
   `TLS=1`, run each one through the container's shell so that the CA path
   expands inside it, for example:

   ```sh
   docker exec falkordb sh -c 'redis-cli --tls --cacert "$FALKORDB_TLS_PATH/ca.crt" SAVE'
   ```

2. **On the Redis image**, remove what Valkey cannot read, and save.

   FalkorDB keeps each graph's query statistics in a stream named
   `telemetry{<graph>}`, and Valkey cannot read streams as Redis saves them.
   Stop FalkorDB from writing new ones, then delete them:

   ```sh
   docker exec falkordb redis-cli GRAPH.CONFIG SET CMD_INFO no
   docker exec falkordb sh -c \
     'redis-cli --scan --pattern "telemetry{*}" | while IFS= read -r key; do redis-cli DEL "$key"; done'
   ```

   `--scan` covers database 0, where FalkorDB keeps its keys. If you use other
   databases, repeat both commands with `redis-cli -n <db>` (on the `--scan`
   and on the `DEL`) for each.

   If you keep other data next to your graphs, delete or convert any other
   stream, any hash with field expiry, and any key of a type only Redis has,
   such as an array, too. This lists the streams:

   ```sh
   docker exec falkordb sh -c \
     'redis-cli --scan | while IFS= read -r key; do [ "$(redis-cli TYPE "$key")" = stream ] && echo "$key"; done'
   ```

   Strings, lists, sets, sorted sets, hashes without field expiry, keys with
   a TTL, and graphs load as they are.

   Then save:

   ```sh
   docker exec falkordb redis-cli SAVE
   ```

   With AOF on, also rewrite the AOF and wait until the rewrite has finished,
   which is when both `aof_rewrite_*` values read `:0` (a rewrite waits,
   `scheduled:1`, while a save runs) and `aof_last_bgrewrite_status` reads `ok`:

   ```sh
   docker exec falkordb redis-cli BGREWRITEAOF
   docker exec falkordb redis-cli INFO persistence | grep -E 'aof_rewrite_(scheduled|in_progress)|aof_last_bgrewrite_status'
   ```

   Stop and remove the container, so that step 3 can reuse its name. The
   data stays in the volume:

   ```sh
   docker stop falkordb && docker rm falkordb
   ```

3. **Start the Valkey image with `FALKORDB_REDIS_MIGRATION=1`** on the same
   volume:

   ```sh
   docker run -d --name falkordb -v <volume>:/var/lib/falkordb/data \
     -e FALKORDB_REDIS_MIGRATION=1 <your other options, VALKEY_ARGS included> \
     falkordb/falkordb-server:<valkey tag>
   ```

   The image starts Valkey with `--rdb-version-check relaxed`, which lets it
   load the Redis files. Check that your graphs are there (`GRAPH.LIST`, and a
   query), then save them in Valkey's format:

   ```sh
   docker exec falkordb redis-cli SAVE
   ```

   With AOF on, also run `BGREWRITEAOF` and wait for it, as in step 2.

   If Valkey stops with `Can't handle foreign type or opcode`, a key it
   cannot read is still in the files. The data is unchanged (the image may have
   changed the files' owner to the `falkordb` user, uid 999; restore the owner
   if you run something else on the volume): remove the container and go back
   to step 2 on the Redis image.

4. **Start the Valkey image again without `FALKORDB_REDIS_MIGRATION`.**
   `docker restart` would keep the variable, so remove the container and run
   it again without `-e FALKORDB_REDIS_MIGRATION=1`:

   ```sh
   docker stop falkordb && docker rm falkordb
   docker run -d --name falkordb -v <volume>:/var/lib/falkordb/data \
     <your other options, VALKEY_ARGS included> \
     falkordb/falkordb-server:<valkey tag>
   ```

   The files are Valkey's now, and the image starts as usual.

Query statistics start again from empty: `GRAPH.CONFIG SET CMD_INFO` lasts
only until the Redis server stops, and Valkey writes new `telemetry{}`
streams in its own format.

## Notes

- Passing `--rdb-version-check relaxed` yourself, in `VALKEY_ARGS` or a
  config file, counts as opting in too. The image prints a notice and lets
  Valkey try to load the files.
- Valkey 9.1.2 does not read an RDB of version 80 or above with a Redis
  header, even with `rdb-version-check relaxed`. The image says so, and the
  volume needs a Redis image.
- Redis images ignore `FALKORDB_REDIS_MIGRATION`.
