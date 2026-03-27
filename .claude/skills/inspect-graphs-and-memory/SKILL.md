---
name: Inspect graphs and memory usage
description: Use GRAPH.LIST, GRAPH.INFO, and GRAPH.MEMORY USAGE to monitor graphs and resources
---

# Inspect graphs and memory usage

Use introspection commands to monitor graph information and resource usage.

## Usage

Use `GRAPH.LIST`, `GRAPH.INFO`, and `GRAPH.MEMORY USAGE` commands for operational visibility.

## Example

    redis-cli GRAPH.LIST
    redis-cli GRAPH.INFO
    redis-cli GRAPH.INFO RunningQueries WaitingQueries
    redis-cli GRAPH.MEMORY USAGE social

## Notes

- `GRAPH.LIST` shows all graphs in the database
- `GRAPH.INFO` is a global command (no graph name argument) that shows running queries, waiting queries, and object pool stats
- `GRAPH.INFO` accepts optional section filters: `RunningQueries`, `WaitingQueries`, `ObjectPool`
- `GRAPH.MEMORY USAGE <graph>` reports memory consumption for a specific graph
- These commands are essential for monitoring and capacity planning
