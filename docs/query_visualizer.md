# Query Visualizer User Guide

## Overview

The Query Visualizer is a terminal-based application for interactively exploring the results of FalkorDB queries. It provides a tree view of query execution steps, allows searching through results, and supports step-by-step navigation.

## Features

- **Query Execution:** Enter and run Cypher queries against a FalkorDB database.
- **Tree Visualization:** View the structure of query results as a tree.
- **Step Navigation:** Move forward and backward through result steps.
- **Search:** Filter steps using custom Python expressions.
- **Query History:** Navigate through previously entered queries.

## Getting Started

### Running the App

1. Ensure FalkorDB server is running and accessible.
2. Run the application:
   ```bash
   python tests/record.py
   ```
3. Optionally, run it as a server
   ```bash
   textual serve tests/record.py
   ```

## Interface Guide

### Main Components

- **Query Label:** Displays the last executed query.
- **Search Input:** Enter a Python expression to filter steps (e.g., `x == 5`).
- **Tree View:** Shows the result structure. Each node represents a step or result.
- **Step Label & Progress Bar:** Indicates the current step and progress.
- **Query Input:** Enter a new Cypher query to execute.

### Controls

#### Navigating Steps

- **Left Arrow:** Move to the previous step.
- **Right Arrow:** Move to the next step.

#### Searching Steps

- Focus the "Search" input (Tab to navigate).
- Enter a Python expression using result variable names (e.g., `x == 3 and y > 5`).
- Press Enter to jump to the next step matching the condition.

#### Running Queries

- Focus the "Enter query" input.
- Type your Cypher query and press Enter to execute.
- Use Up/Down arrows to navigate query history.

#### Tree Interaction

- The tree updates to highlight the current step.
- The label shows the environment (i.e., variable values) for the selected step.

## Example Usage

1. Start the app.
2. Enter a query like:
   ```
   UNWIND range(1, 10) AS x UNWIND range(x, 10) AS y RETURN x, y
   ```
3. Use the arrow keys to step through results.
4. Use the search box to filter steps (e.g., `x == 5`).

## Notes

- The search uses Python's `eval()`; use valid expressions and variable names as shown in the tree.
- The app maintains a history of queries for easy recall.