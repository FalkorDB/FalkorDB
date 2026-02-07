---
name: Match patterns and return projections
description: Match nodes by label and properties using MATCH, then return specific fields from query results
---

# Match patterns and return projections

Match nodes by label and properties, then return specific fields from the query results.

## Usage

Use `MATCH` to find nodes and relationships based on patterns, then use `RETURN` to project specific properties.

## Example

    redis-cli GRAPH.QUERY social "MATCH (alice:User {name: 'Alice'})-[:FRIENDS_WITH]->(friend)
    RETURN friend.name"

## Notes

- `MATCH` accepts label filters (`:User`) and property filters (`{name: 'Alice'}`)
- Relationship patterns are specified between nodes using `-[:RELATIONSHIP_TYPE]->`
- `RETURN` allows you to select specific properties to return from matched nodes
- Use dot notation to access node properties (e.g., `friend.name`)
