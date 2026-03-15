#!/usr/bin/env bash
set -euo pipefail

AUTOSTART_BIN="${FERRITE_AUTOSTART_BIN:-$HOME/.local/bin/ferrite-autostart}"
URL_FILE="${FERRITE_REMOTE_URL_FILE:-$HOME/.local/share/ferrite/remote-session-url.txt}"
LOG_FILE="${FERRITE_AUTOSTART_LOG:-$HOME/.local/share/ferrite/autostart.log}"
TAILSCALE_ACCEPT_ARGS="${FERRITE_TAILSCALE_ARGS:---accept-routes}"

info() { printf '[ferrite-up] %s\n' "$*"; }
warn() { printf '[ferrite-up] WARN: %s\n' "$*" >&2; }
die() { printf '[ferrite-up] ERROR: %s\n' "$*" >&2; exit 1; }

ensure_tailscale() {
  if ! command -v tailscale >/dev/null 2>&1; then
    warn "tailscale not found; skipping network bring-up"
    return 0
  fi
  if tailscale status >/dev/null 2>&1; then
    info "tailscale already running"
    return 0
  fi
  info "bringing up tailscale"
  if sudo -n tailscale up ${TAILSCALE_ACCEPT_ARGS} >/dev/null 2>&1; then
    sleep 3
    return 0
  fi
  if tailscale up ${TAILSCALE_ACCEPT_ARGS} >/dev/null 2>&1; then
    sleep 3
    return 0
  fi
  warn "tailscale up failed; SSH may be unavailable"
}

start_autostart() {
  [[ -x "$AUTOSTART_BIN" ]] || die "missing $AUTOSTART_BIN; reinstall feRcuda or copy scripts/ferrite-autostart into ~/.local/bin"
  mkdir -p "$(dirname "$LOG_FILE")"
  if command -v systemctl >/dev/null 2>&1 && systemctl --user status ferrite-session >/dev/null 2>&1; then
    info "starting systemd user unit ferrite-session"
    systemctl --user start ferrite-session
    return 0
  fi
  if pgrep -af "ferrite-autostart" >/dev/null 2>&1; then
    info "ferrite-autostart already running"
    return 0
  fi
  info "spawning ferrite-autostart directly"
  nohup "$AUTOSTART_BIN" >>"$LOG_FILE" 2>&1 &
}

show_status() {
  if [[ -f "$URL_FILE" ]]; then
    info "saved URL: $(head -n 1 "$URL_FILE")"
  else
    info "saved URL: pending"
  fi
  info "attach tmux: tmux attach -t ${FERRITE_SESSION:-claude-remote}"
  info "log file: $LOG_FILE"
}

ensure_tailscale
start_autostart
show_status
