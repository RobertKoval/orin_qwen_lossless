#!/usr/bin/env bash
# Log memory every second and kill a runaway server before the board freezes (it froze once at 0 available memory).
#   tools/orin/mem_watch.sh <pidfile> <log> [min_mib=1536]   (run as root to read the nvmap pool)
# Logs: time, MemAvailable, MemFree, Cached (MiB), nvmap page pool (MiB, if readable), RSS of the pid tree (MiB).
# Kills the process tree (TERM, then KILL after 5 s) when MemFree + Cached + nvmap pool < min_mib. (MemAvailable
# is ~6 GB pessimistic here: it subtracts the zone watermarks and ignores the nvmap pool.)
set -u
pidfile=$1; log=$2; min=${3:-1536}
pool_file=/sys/kernel/debug/nvmap/pagepool/page_pool_available_pages
tree() { local p=$1; echo "$p"; for c in $(cat /proc/"$p"/task/*/children 2>/dev/null); do tree "$c"; done; }
echo "# t avail_mib free_mib cached_mib pool_mib rss_mib" > "$log"
peak_used=0
seen=""  # every descendant ever seen: vLLM's engine core is reparented to init when the API server dies
while sleep 1; do
    pid=$(cat "$pidfile" 2>/dev/null) || continue
    alive=0; for p in $pid $seen; do kill -0 "$p" 2>/dev/null && alive=1; done
    [ $alive = 1 ] || { echo "# server gone, peak used $peak_used MiB" >> "$log"; exit 0; }
    read -r avail free cached < <(awk '/MemAvailable/{a=$2}/MemFree/{f=$2}/^Cached/{c=$2}END{print int(a/1024),int(f/1024),int(c/1024)}' /proc/meminfo)
    pool=$(( $(cat $pool_file 2>/dev/null || echo 0) * 4 / 1024 ))
    for p in $(tree "$pid"); do case " $seen " in *" $p "*) ;; *) seen="$seen $p";; esac; done
    rss=0; for p in $seen; do r=$(awk '/VmRSS/{print $2}' /proc/"$p"/status 2>/dev/null || echo 0); rss=$((rss + ${r:-0})); done
    used=$(( $(awk '/MemTotal/{print int($2/1024)}' /proc/meminfo) - free - cached - pool ))
    [ $used -gt $peak_used ] && peak_used=$used
    echo "$(date +%T) $avail $free $cached $pool $((rss / 1024)) used=$used" >> "$log"
    if [ $((free + cached + pool)) -lt "$min" ]; then
        echo "# KILL: free $free + cached $cached + pool $pool MiB < $min" >> "$log"
        for p in $seen; do kill -TERM "$p" 2>/dev/null; done
        sleep 5
        for p in $seen; do kill -KILL "$p" 2>/dev/null; done
        exit 1
    fi
done
