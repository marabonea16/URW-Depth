#!/bin/bash
# Monitorizeaza antrenarea si reia daca s-a oprit.
# Rulat de cron la fiecare ora.

set -euo pipefail

WORKDIR="/home/ubuntu/TinyDepth"
LOG="/home/ubuntu/monitor_training.log"
HC_URL="https://hc-ping.com/f2d24240-e53d-4315-bf2d-c9bae9778766"

cd "$WORKDIR"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

# Ping healthchecks.io so it knows the VM is alive
wget -qO- "$HC_URL" > /dev/null 2>&1 || true

# -------------------------------------------------------
# URW-Depth-S2-Fix2/Fix3: ABANDONATE (regresie/divergenta).
# Rulam acum un diagnostic MANUAL scurt (URW-Depth-Calib-Diag),
# nu via monitor. Auto-resume dezactivat temporar ca sa nu
# coliziune cu rularile manuale de diagnostic.
# -------------------------------------------------------

echo "$(timestamp) Auto-resume dezactivat (diagnostic manual in curs)." >> "$LOG"
exit 0
