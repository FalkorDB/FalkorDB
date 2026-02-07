---
name: Track slow queries
description: Use GRAPH.SLOWLOG to identify and monitor slow queries and performance bottlenecks
---

# Track slow queries

Use the slowlog to identify and monitor queries that take longer than expected.

## Usage

Use `GRAPH.SLOWLOG` to view slow queries and `GRAPH.SLOWLOG RESET` to clear the log.

## Example

    redis-cli GRAPH.SLOWLOG social
    redis-cli GRAPH.SLOWLOG social RESET

## Notes

- The slowlog captures queries that exceed a configured time threshold
- Helps identify performance bottlenecks and optimization opportunities
- Use the slowlog during development and in production monitoring
- Reset the slowlog after addressing performance issues to track new ones
- Essential tool for maintaining query performance over time
