# FalkorDB Docker Compose

This directory contains a Docker Compose configuration for running FalkorDB with a browser interface.

## Services

- **falkordb-server**: The FalkorDB graph database server (port 6379)
- **falkordb-browser**: Web-based browser interface for FalkorDB (port 3000)

## Getting Started

### Prerequisites

- Docker
- Docker Compose

### Starting the Services

To start both FalkorDB server and browser, run:

```bash
docker compose up -d
```

This will:
- Start the FalkorDB server on port 6379
- Start the FalkorDB browser on port 3000
- Create a persistent volume for data storage

### Accessing the Browser

1. Open your web browser and navigate to: `http://localhost:3000`
2. In the browser interface, use the following connection settings:
   - **Hostname**: `host.docker.internal`
   - **Port**: `6379`

**Note**: The hostname `host.docker.internal` is required to connect from the browser container to the FalkorDB server service.

### Stopping the Services

To stop the services:

```bash
docker compose down
```

To stop the services and remove the data volume:

```bash
docker compose down -v
```

## Configuration

- **falkordb.conf**: Configuration file for FalkorDB server
- **docker-compose.yml**: Docker Compose configuration

### Data Persistence

Data is persisted in a Docker volume named `falkordb-data`. This ensures your graph data survives container restarts.
