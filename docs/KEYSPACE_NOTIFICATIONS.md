# Redis Keyspace Notifications for FalkorDB

## Overview

FalkorDB now emits Redis keyspace notifications when graph data is modified. This enables other Redis modules and applications to react to graph changes in real-time without polling.

## Notification Types

The following keyspace notifications are emitted by FalkorDB:

### Graph Operations
- `graph.create` - When a new graph is created
- `graph.delete` - When a graph is deleted/dropped
- `graph.query` - When a write query is executed (emitted alongside specific node/edge events)

### Node Operations
- `graph.node.create` - When nodes are created
- `graph.node.update` - When node properties or labels are modified
- `graph.node.delete` - When nodes are deleted

### Edge/Relationship Operations
- `graph.edge.create` - When edges/relationships are created
- `graph.edge.update` - When edge properties are modified
- `graph.edge.delete` - When edges are deleted

### Index Operations
- `graph.index.create` - When an index is created
- `graph.index.drop` - When an index is dropped

### Constraint Operations
- `graph.constraint.create` - When constraints are added
- `graph.constraint.drop` - When constraints are removed

## Configuration

These notifications only fire when Redis keyspace notifications are enabled. To enable module notifications, use:

```
CONFIG SET notify-keyspace-events AKd
```

Where:
- `A` - Alias for "g$lshzxed", enables all events except `m` (key miss)
- `K` - Keyspace events published as `__keyspace@<db>__:<key>`
- `d` - Module key space notification

For development/testing, you can use:
```
CONFIG SET notify-keyspace-events AKEd
```

Where `E` adds keyevent notifications published as `__keyevent@<db>__:<event>`.

## Subscribing to Events

### From Redis CLI

Subscribe to all graph events for a specific key:
```
PSUBSCRIBE __keyspace@0__:mygraph
```

Subscribe to all graph events across all keys:
```
PSUBSCRIBE __keyspace@0__:*
```

Subscribe to a specific event type:
```
PSUBSCRIBE __keyevent@0__:graph.node.create
```

### From Python

```python
import redis

# Connect to Redis
r = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Enable notifications
r.config_set('notify-keyspace-events', 'AKd')

# Create a pub/sub instance
pubsub = r.pubsub()

# Subscribe to all graph keyspace events
pubsub.psubscribe('__keyspace@0__:*')

# Listen for messages
for message in pubsub.listen():
    if message['type'] == 'pmessage':
        print(f"Key: {message['channel']}, Event: {message['data']}")
```

### From a Redis Module

```c
int GraphChangeCallback(RedisModuleCtx *ctx, int type, const char *event, 
                        RedisModuleString *key) {
    if (strcmp(event, "graph.query") == 0) {
        // React to graph modification
        const char *keyname = RedisModule_StringPtrLen(key, NULL);
        // ... handle the change ...
    }
    return REDISMODULE_OK;
}

// In module initialization
RedisModule_SubscribeToKeyspaceEvents(ctx, 
    REDISMODULE_NOTIFY_MODULE,
    GraphChangeCallback);
```

## Implementation Details

- Notifications are emitted using `RedisModule_NotifyKeyspaceEvent()` with `REDISMODULE_NOTIFY_MODULE` type
- Notifications are emitted after the operation completes successfully
- Failed operations do not emit notifications
- Notifications follow Redis's standard keyspace notification mechanism
- There is no performance impact when keyspace notifications are disabled

## Performance Considerations

- Keyspace notifications have minimal overhead when enabled
- Notifications are only sent to subscribers (no overhead if no subscribers exist)
- When disabled (default), there is zero performance impact
- The module automatically detects notification settings via Redis configuration

## Known Limitations

- **Property update events**: When properties are modified on nodes or edges, both `graph.node.update` and `graph.edge.update` events are emitted because the result statistics don't distinguish between node and edge property updates. This is a conservative approach that may produce false positives (e.g., emitting `graph.edge.update` when only node properties were modified).

## Use Cases

1. **Event-driven architectures** - Build reactive systems that respond to graph changes
2. **Cache invalidation** - Invalidate caches when graph data changes
3. **Audit logging** - Track all modifications to graph data
4. **Change data capture** - Replicate graph changes to other systems
5. **Module interoperability** - Allow other Redis modules to react to FalkorDB changes
6. **Real-time analytics** - Trigger analytics pipelines when data changes
7. **Webhooks** - Notify external services of graph modifications

## Testing

Tests for keyspace notifications are located in:
```
tests/flow/test_keyspace_notifications.py
```

Run the tests with:
```bash
cd tests/flow
python3 -m pytest test_keyspace_notifications.py -v
```

## Compatibility

- Requires Redis 6.0 or later (for module keyspace notifications)
- Compatible with all FalkorDB commands that modify graph data
- Works with both single-node and clustered Redis deployments
