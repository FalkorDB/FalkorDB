---
name: Inspect query plans before execution
description: Use GRAPH.EXPLAIN to view query execution plans and validate index usage without running
---

# Inspect query plans before execution

Use `GRAPH.EXPLAIN` to view the query execution plan without running the query.

## Usage

Use `GRAPH.EXPLAIN` to validate query plan structure and index usage before executing potentially expensive queries.

## Example

    redis-cli GRAPH.EXPLAIN social "MATCH (p:Person {age: 30}) RETURN p"

## Notes

- `GRAPH.EXPLAIN` shows the execution plan without actually running the query
- Helps identify if indexes are being used effectively
- Useful for query optimization and debugging performance issues
- Shows the logical operators that will be used to execute the query
- Does not return actual data, only the execution plan
