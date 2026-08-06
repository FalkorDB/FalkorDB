# Snapcraft Store Setup for FalkorDB

This document describes how to set up the Snapcraft store credentials for automated publishing.

## Prerequisites

1. A Snapcraft account (create at https://snapcraft.io/)
2. Register the `falkordb` snap name in the Snapcraft store
3. Access to the FalkorDB GitHub repository settings

## Steps to Register the Snap

### 1. Install Snapcraft

```bash
sudo snap install snapcraft --classic
```

### 2. Login to Snapcraft

```bash
snapcraft login
```

### 3. Register the Snap Name

```bash
snapcraft register falkordb
```

Note: Snap names are registered on a first-come, first-served basis.

### 4. Export Snapcraft Credentials

```bash
snapcraft export-login snapcraft-credentials.txt
```

This creates a base64-encoded credentials file.

### 5. Add Credentials to GitHub Secrets

1. Go to the FalkorDB repository settings
2. Navigate to **Settings** > **Secrets and variables** > **Actions**
3. Click **New repository secret**
4. Name: `SNAPCRAFT_STORE_CREDENTIALS`
5. Value: Paste the entire content of `snapcraft-credentials.txt`
6. Click **Add secret**

**Important**: Delete `snapcraft-credentials.txt` after adding it to GitHub secrets.

## Testing the Workflow

After setting up the credentials, the workflow will automatically:

- Build snaps on pull requests (without publishing)
- Publish to the **edge** channel when code is pushed to the `master` branch
- Publish to **stable** and **candidate** channels when a version tag is pushed (e.g., `v4.2.1`)

### Manual Trigger

You can also manually trigger the workflow:

1. Go to **Actions** tab in the GitHub repository
2. Select the "Build and Publish Snap" workflow
3. Click **Run workflow**
4. Select the branch to run on

## Snap Store Channels

The snap uses the following channel strategy:

- **stable**: Released versions (tags like `v4.2.1`)
- **candidate**: Pre-release testing (same as stable)
- **edge**: Latest development builds (from `master` branch)
- **beta**: Not currently used

Users can install from different channels:
```bash
# Stable (recommended for production)
sudo snap install falkordb

# Edge (latest development)
sudo snap install falkordb --edge
```

## Monitoring Releases

Track snap releases at:
- Store dashboard: https://snapcraft.io/falkordb/releases
- Metrics: https://snapcraft.io/falkordb/metrics

## Troubleshooting

### Publishing fails with "Invalid credentials"

Re-export and update the GitHub secret:
```bash
snapcraft export-login snapcraft-credentials.txt
# Then update the GitHub secret with the new content
```

### Snap name already registered

If someone else has registered the name, you'll need to:
1. Contact Snapcraft support to request a name transfer
2. Or choose a different name in `snapcraft.yaml`

### Architecture build failures

Ensure both architectures are registered:
```bash
snapcraft list-registered
```

If an architecture is missing, builds for that platform will be skipped.

## Security Notes

- Never commit snapcraft credentials to the repository
- Rotate credentials periodically for security
- Use repository secrets, not environment variables in workflows
- Limit access to the repository settings to authorized users only

## Additional Resources

- [Snapcraft Documentation](https://snapcraft.io/docs)
- [GitHub Actions Snapcraft Publishing](https://snapcraft.io/docs/build-on-github)
- [Snapcraft Store Metrics](https://snapcraft.io/docs/store-metrics)
