# FalkorDB Snap Package

This directory contains the configuration for building and publishing FalkorDB as a snap package.

## Overview

The FalkorDB snap provides an easy way to install and run FalkorDB on Linux systems. The snap includes:
- Redis server (version 8.2.2)
- FalkorDB module
- All necessary dependencies

## Installation

### From Snap Store (when published)

For stable releases:
```bash
sudo snap install falkordb
```

For the latest development version (edge channel):
```bash
sudo snap install falkordb --edge
```

### Manual Installation

To install a locally built snap:
```bash
sudo snap install falkordb_*.snap --dangerous
```

## Usage

After installation, the FalkorDB service starts automatically. You can manage it with:

```bash
# Check service status
sudo snap services falkordb

# Restart the service
sudo snap restart falkordb

# Stop the service
sudo snap stop falkordb

# Start the service
sudo snap start falkordb

# View logs
sudo snap logs falkordb
```

### Connect to FalkorDB

You can connect to the running FalkorDB instance using redis-cli or any Redis client:

```bash
redis-cli
```

### Data Persistence

Data is stored in `/var/snap/falkordb/common/data` and persists across snap refreshes and reboots.

## Building the Snap

### Prerequisites

Install snapcraft:
```bash
sudo snap install snapcraft --classic
```

### Build

From the repository root:
```bash
snapcraft
```

For a specific architecture:
```bash
snapcraft --target-arch=arm64
```

### Clean

To clean the build environment:
```bash
snapcraft clean
```

## Snap Configuration

The snap is configured with:
- **Confinement**: `strict` - Maximum security isolation
- **Base**: `core24` - Ubuntu 24.04 LTS base
- **Architectures**: amd64, arm64
- **Network access**: Required for Redis server operation
- **Data directory**: `$SNAP_COMMON/data` (persistent across updates)

## CI/CD

The snap is automatically built and published via GitHub Actions:
- **Edge channel**: Automatically published on pushes to `master` branch
- **Stable/Candidate channels**: Automatically published on version tags (e.g., `v4.2.1`)

## Troubleshooting

### Service won't start

Check the logs:
```bash
sudo snap logs falkordb -f
```

### Connection refused

Verify the service is running:
```bash
sudo snap services falkordb
```

### Port already in use

The snap uses port 6379 by default. If another Redis instance is running, stop it first:
```bash
sudo systemctl stop redis-server
```

## Development

To test local changes:

1. Make your changes to `snapcraft.yaml`
2. Build the snap: `snapcraft`
3. Install locally: `sudo snap install falkordb_*.snap --dangerous`
4. Test the installation

## Links

- [Snapcraft Documentation](https://snapcraft.io/docs)
- [FalkorDB Documentation](https://docs.falkordb.com/)
- [FalkorDB Repository](https://github.com/FalkorDB/FalkorDB)
