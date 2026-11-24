#!/usr/bin/env bash
# Run a command with a proxy reachable from this user's Docker build and setup phases.
set -euo pipefail
unset DOCKER_CONTEXT
uid=$(id -u)
export DOCKER_HOST="unix:///run/user/$uid/experiment-docker.sock"
test -S "${DOCKER_HOST#unix://}"
child=$(cat "/run/user/$uid/experiment-rootlesskit/child_pid")
relay="$HOME/.local/lib/experiment-docker/proxy_relay.py"
relay_socket="$HOME/container-data/proxy/http.sock"
test -f "$relay" && test -S "$relay_socket"
bind_ip=$(nsenter -t "$child" -U -n -- ip -4 -o addr show tap0 | awk '{split($4,a,"/"); print a[1]}')
test -n "$bind_ip"
port=${EVOEFFICIENT_PROXY_PORT:-17892}
# Prevent two wrappers in this user session from owning the same endpoint.
exec 9>"/run/user/$uid/evoefficient-proxy-$port.lock"
flock -n 9 || { echo "实验容器代理端口 $port 正在使用" >&2; exit 1; }
nsenter -t "$child" -U -n -- /usr/bin/python3 "$relay" \
  --listen-address "$bind_ip" --listen-port "$port" --target-unix "$relay_socket" &
relay_pid=$!
trap 'kill "$relay_pid" 2>/dev/null || true; wait "$relay_pid" 2>/dev/null || true' EXIT
for attempt in {1..25}; do
  kill -0 "$relay_pid"
  if nsenter -t "$child" -U -n -- ss -ltnH "sport = :$port" | grep -Fq "$bind_ip:$port"; then
    export EVOEFFICIENT_CONTAINER_PROXY="http://$bind_ip:$port"
    "$@"
    exit
  fi
  sleep .1
done
echo '实验容器代理未启动' >&2
exit 1
