---
name: Use MERGE to avoid duplicate nodes
description: Use MERGE for idempotent upserts with ON CREATE and ON MATCH clauses to avoid duplicates
---

# Use MERGE to avoid duplicate nodes

Use `MERGE` for idempotent upserts that create nodes only if they don't exist.

## Usage

Use `MERGE` when you want to ensure a node exists without creating duplicates. Combine with `ON CREATE` and `ON MATCH` clauses to set properties conditionally.

## Example

    redis-cli GRAPH.QUERY social "MERGE (u:User {id: 42})
    ON CREATE SET u.name = 'Dana'
    ON MATCH SET u.last_seen = timestamp()"

## Notes

- `MERGE` creates the node if it doesn't exist, or matches it if it does
- `ON CREATE SET` runs only when a new node is created
- `ON MATCH SET` runs only when an existing node is matched
- This ensures idempotent operations that can be safely repeated
