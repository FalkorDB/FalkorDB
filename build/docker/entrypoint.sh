#!/bin/sh

# Convert CYPHER environment variable to boolean for supervisord
case "${CYPHER:-}" in
    1|true|True|TRUE)
        export CYPHER="true"
        echo "text-to-cypher mode enabled"
        ;;
    *)
        export CYPHER="false"
        echo "text-to-cypher mode disabled"
        ;;
esac

# Convert BROWSER environment variable to boolean for supervisord
case "${BROWSER:-}" in
    1|true|True|TRUE)
        export BROWSER="true"
        echo "FalkorDB Browser enabled"
        ;;
    *)
        export BROWSER="false"
        echo "FalkorDB Browser disabled"
        ;;
esac



# Start supervisord
if ! command -v /usr/bin/supervisord >/dev/null 2>&1; then
    echo "Error: supervisord not found in PATH!"
    exit 1
else
    echo "Starting supervisord..."
fi


exec /usr/bin/supervisord -n -c /etc/supervisor/conf.d/supervisord.conf