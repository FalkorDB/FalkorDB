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
    redis-cli GRAPH.INFO social
    redis-cli GRAPH.MEMORY USAGE social

## Notes

- `GRAPH.LIST` shows all graphs in the database
- `GRAPH.INFO` provides statistics about a specific graph (node count, edge count, etc.)
- `GRAPH.MEMORY USAGE` reports memory consumption for a graph
- These commands are essential for monitoring and capacity planning
- Use them to understand resource utilization and graph characteristics
