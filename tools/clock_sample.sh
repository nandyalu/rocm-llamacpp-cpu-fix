#!/bin/bash
# Sample every AMD card once a second: time, card number, busy %, memory clock, core clock, power (uW).
# Healthy generation: busy 95-99, mclk 875 MHz, 80-100 W. A stalled card showed busy 49, mclk 96 MHz, 18-23 W.
# Needs no root. Stop with Ctrl-C.
while :; do
  t=$(date -u +%H:%M:%S)
  for c in /sys/class/drm/card[0-9]*/device; do
    [ -f $c/pp_dpm_mclk ] || continue
    n=${c#/sys/class/drm/card}; n=${n%%/*}
    echo "$t $n $(cat $c/gpu_busy_percent) $(grep '\*' $c/pp_dpm_mclk | awk '{print $2}') $(grep '\*' $c/pp_dpm_sclk | awk '{print $2}') $(cat $c/hwmon/hwmon*/power1_average)"
  done
  timeout 1 tail -f /dev/null
done
