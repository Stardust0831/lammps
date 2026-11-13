#!/usr/bin/env bash
set -euo pipefail

case "${1:---prepare}" in
    --prepare|--install) mode=${1:---prepare} ;;
    *) echo 'Usage: install.sh [--prepare|--install]' >&2; exit 2 ;;
esac
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
umask 077
mkdir -p "$HOME/.local/lib/lammps-sai-scheduler" \
    "$HOME/.local/state/lammps-sai-scheduler" "$HOME/.config/systemd/user"
install -m 644 "$source_dir/dispatch.py" "$HOME/.local/lib/lammps-sai-scheduler/dispatch.py"
for name in lammps-sai-scheduler.service lammps-sai-scheduler.timer; do
    install -m 644 "$source_dir/$name" "$HOME/.config/systemd/user/$name"
done
systemctl --user daemon-reload
if [ "$mode" = --prepare ]; then
    echo 'Scheduler files prepared; timer enablement is unchanged.'
    exit 0
fi
systemd-analyze --user verify "$HOME/.config/systemd/user/lammps-sai-scheduler.service" \
    "$HOME/.config/systemd/user/lammps-sai-scheduler.timer"
systemctl --user enable --now lammps-sai-scheduler.timer
systemctl --user start lammps-sai-scheduler.service
systemctl --user list-timers lammps-sai-scheduler.timer --no-pager
