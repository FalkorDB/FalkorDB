---
name: Create and query vector indexes
description: Use HNSW vector indexes for Approximate Nearest Neighbor (ANN) search with embeddings
---

# Create and query vector indexes

Use HNSW vector indexes for Approximate Nearest Neighbor (ANN) search.

## Usage

Create vector indexes with specific dimension and similarity configurations, then query them using `db.idx.vector.queryNodes`.

## Example

    redis-cli GRAPH.QUERY social "CREATE VECTOR INDEX FOR (p:Product) ON (p.embedding)
    OPTIONS {dimension: 768, similarityFunction: 'cosine', M: 32, efConstruction: 200}"

    redis-cli GRAPH.QUERY social "CALL db.idx.vector.queryNodes('Product', 'embedding', 5, vecf32([0.1, 0.2, 0.3]))
    YIELD node, score RETURN node.name, score"

## Notes

- Vector indexes use HNSW (Hierarchical Navigable Small World) algorithm
- Configure dimension to match your embedding size
- Similarity functions include 'cosine', 'euclidean', etc.
- `M` and `efConstruction` parameters tune index performance and accuracy
- Use `vecf32()` to pass vector values in queries
- Returns nodes with their similarity scores
