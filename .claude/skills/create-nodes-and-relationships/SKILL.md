---
name: Create nodes and relationships
description: Create labeled nodes and connect them with properties using CREATE in FalkorDB
---

# Create nodes and relationships

Create labeled nodes and connect them with properties in FalkorDB.

## Usage

Use `CREATE` to add nodes with labels and properties, and establish relationships between them.

## Example

    redis-cli GRAPH.QUERY social "CREATE (alice:User {id: 1, name: 'Alice', email: 'alice@example.com'})
    CREATE (bob:User {id: 2, name: 'Bob', email: 'bob@example.com'})
    CREATE (alice)-[:FRIENDS_WITH {since: 1640995200}]->(bob)"

## Notes

- Nodes are labeled with `:Label` syntax (e.g., `:User`)
- Properties are defined in curly braces using JSON-like syntax
- Relationships are created with `->` syntax and can have labels and properties
- All commands use `GRAPH.QUERY` with the graph name as the first parameter
