- [Security Policy](#security-policy)
  - [Supported Versions](#supported-versions)
  - [Reporting a Vulnerability](#reporting-a-vulnerability)

# Security Policy

This Information Security Policy describes the security standards and practices followed by FalkorDB in the development and distribution of  FalkorDB (the “Software”).

While users of the Software are responsible for securing their own deployments, FalkorDB implements the following security practices to help ensure that the Software is released in a secure, reliable, and trustworthy manner:
1. Secure Software Development

    FalkorDB follows secure software development lifecycle (SDLC) practices, including code reviews, automated testing, and security scanning.
    All code contributions to FalkorDB’s repositories are reviewed and must pass CI pipelines that include security-focused tests.
    Critical dependencies are regularly reviewed for vulnerabilities and updated as needed.

2. Vulnerability Management

    Known vulnerabilities in the Software or its dependencies are tracked and remediated as quickly as possible.
    FalkorDB maintains a responsible disclosure process for reporting security vulnerabilities (e.g., via a security email or GitHub Security Advisories).
    Security fixes are released publicly through tagged releases and documented in release notes.

3. Access and Change Control

    Only authorized maintainers have write access to the source code repositories and release pipelines.
    All changes to the codebase are auditable through version control (e.g., GitHub).

4. Build Integrity and Distribution

    Official releases of the Software are built using controlled CI/CD systems.
    Release artifacts (e.g., source archives, Docker images) are signed or checksummed to ensure integrity and authenticity.

5. Third-Party Components

    The Software may include third-party open-source dependencies, which are scanned for known vulnerabilities using industry-standard tools.
    FalkorDB maintains an internal software bill of materials (SBOM) and will respond to CVEs in third-party components used in the Software.

6. Security Communication

    Users are encouraged to subscribe to FalkorDB’s release channels (e.g., GitHub releases, mailing list, website) for notifications of security updates.
    Security issues should be reported responsibly via the published disclosure process.

7. Limitations

    This policy covers only the security of the Software as it is released by FalkorDB.
    FalkorDB is not responsible for securing user deployments or environments. Users are solely responsible for configuring, operating, and maintaining secure deployments of the Software.

## Supported Versions

We release patches for security vulnerabilities. Which versions are eligible for
receiving such patches depends on the CVSS v3.0 Rating:

| CVSS v3.0 | Supported Versions                        |
| --------- | ----------------------------------------- |
| X.Y.Z     | Releases within the previous three months |
| X.Y.Z     | Most recent release                       |

## Reporting a Vulnerability

Please report (suspected) security vulnerabilities to
**[security@falkordb.com](mailto:security@falkordb.com)**. You will receive a response from
us within 48 hours. If the issue is confirmed, we will release a patch as soon
as possible depending on complexity but historically within a few days.
