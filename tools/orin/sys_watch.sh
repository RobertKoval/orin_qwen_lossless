#!/usr/bin/env bash
# Background system sampler for benchmark sessions: vmstat (swap in/out, CPU) and tegrastats (clocks, EMC, temps)
# every second until killed.   tools/orin/sys_watch.sh <prefix>  -> <prefix>-vmstat.log, <prefix>-tegra.log
vmstat -t 1 > "$1-vmstat.log" &
tegrastats --interval 1000 > "$1-tegra.log" &
wait
