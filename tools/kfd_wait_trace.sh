#!/usr/bin/env bash
# Trace every AMDKFD wait (kfd_wait_on_events) of all processes. Run as root, with bpftrace installed.
# Every 5 s it prints, for each (pid, tid): the number of calls, the last timeout, the number of
# events, the first event id, the event age on the way in and out, and the last result.
# A thread in a busy loop shows thousands of calls a second, and age in != age out.
# Stop with Ctrl-C. Usage: sudo ./kfd_wait_trace.sh > kfd_waits.txt
set -uo pipefail
# struct kfd_event_data is 48 bytes: last_event_age at 0, event_id at 40.
exec bpftrace -e '
fentry:amdgpu:kfd_wait_on_events {
  $d = (uint64)args.data;
  @calls[pid, tid] = count();
  @timeout_ms[pid, tid] = *args.user_timeout_ms;
  @nev[pid, tid] = args.num_events;
  @ev0[pid, tid] = *(uint32 *)uptr($d + 40);
  @age_in[pid, tid] = *(uint64 *)uptr($d);
}
fexit:amdgpu:kfd_wait_on_events {
  $d = (uint64)args.data;
  @result[pid, tid] = *args.wait_result;
  @age_out[pid, tid] = *(uint64 *)uptr($d);
}
interval:s:5 {
  printf("== now %llu\n", nsecs);
  print(@calls); print(@timeout_ms); print(@nev); print(@ev0);
  print(@age_in); print(@age_out); print(@result);
  clear(@calls);
}'
