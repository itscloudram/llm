#!/usr/bin/env bash
# reserve-gpu.sh - keep trying to reserve ONE GPU instance (g6e.2xlarge or g7e.2xlarge; Linux/UNIX, open,
# default tenancy) in the listed Availability Zones until AWS grants it. Stops on the first success.
#
# Cost: failed attempts cost nothing (they show as "Failed" in the console for 60 min). An active reservation bills at the On-Demand rate
# (g7e.2xlarge $3.363/hr, g6e.2xlarge $2.242/hr) until YOU cancel it (end date = Manually); while your instance runs inside it
# you pay only the instance.
#
# Usage (options go on the SAME line, before "bash"):
#   TYPE=g6e.2xlarge bash reserve-gpu.sh
#   TYPE=g7e.2xlarge AZS="us-east-1a us-east-1c" bash reserve-gpu.sh
# Options (environment variables):
#   TYPE=g6e.2xlarge       instance type to reserve (default g6e.2xlarge)
#   AZS="us-east-1a us-east-1b"  zones to try, in order (default: 1a 1b 1c 1d; zones that do not offer TYPE are skipped)
#   INTERVAL=60            seconds between rounds (each round tries every zone once)
#   SNS_TOPIC_ARN=arn:...  optional: publish a message when the reservation is granted
set -uo pipefail

REGION="us-east-1"
TYPE="${TYPE:-g6e.2xlarge}"
read -r -a AZS <<< "${AZS:-us-east-1a us-east-1b us-east-1c us-east-1d}"
INTERVAL="${INTERVAL:-60}"
SNS_TOPIC_ARN="${SNS_TOPIC_ARN:-}"
LOG="reserve-gpu.log"

log() { printf '%s  %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG"; }

# 0. Pre-flight: credentials work, and no active reservation already exists (avoids paying twice).
if ! aws sts get-caller-identity --region "$REGION" >/dev/null 2>&1; then
  log "ERROR: AWS credentials not available. Run this in CloudShell or after 'aws configure'."
  exit 1
fi
existing=$(aws ec2 describe-capacity-reservations --region "$REGION" \
  --filters "Name=instance-type,Values=$TYPE" "Name=state,Values=active,pending" \
  --query 'CapacityReservations[].[CapacityReservationId,AvailabilityZone,State]' --output text 2>&1) || {
  log "ERROR: cannot list capacity reservations: $existing"; exit 1; }
if [ -n "$existing" ]; then
  log "A $TYPE reservation already exists - not creating another:"
  log "$existing"
  exit 0
fi

# 1. Keep only zones where AWS offers this type at all (g7e.2xlarge, for example, is not in every AZ).
offered=$(aws ec2 describe-instance-type-offerings --region "$REGION" --location-type availability-zone \
  --filters "Name=instance-type,Values=$TYPE" --query 'InstanceTypeOfferings[].Location' --output text 2>&1) || {
  log "ERROR: cannot list where $TYPE is offered: $offered"; exit 1; }
keep=()
for az in "${AZS[@]}"; do
  if [[ " $offered " == *" $az "* ]] || [[ "$offered" == *"$az"* ]]; then keep+=("$az"); else log "skipping $az: $TYPE is not offered there"; fi
done
AZS=("${keep[@]}")
if [ "${#AZS[@]}" -eq 0 ]; then
  log "ERROR: $TYPE is not offered in any requested zone. AWS offers it in: ${offered:-none}"
  exit 1
fi

# 2. A created reservation can still end up 'failed' (no capacity) a moment later - wait for the real state.
wait_state() {  # prints active | failed | pending
  local id=$1 st=""
  for _ in $(seq 1 30); do
    st=$(aws ec2 describe-capacity-reservations --region "$REGION" --capacity-reservation-ids "$id" \
      --query 'CapacityReservations[0].State' --output text 2>/dev/null)
    case "$st" in active|failed|cancelled|expired) break ;; esac
    sleep "${STATE_POLL:-10}"
  done
  [ "$st" = cancelled ] || [ "$st" = expired ] && st=failed
  echo "${st:-pending}"
}

log "Trying to reserve 1x $TYPE in ${AZS[*]} every ${INTERVAL}s until granted. Ctrl+C to stop."
round=0
while true; do
  round=$((round + 1))
  for az in "${AZS[@]}"; do
    if out=$(aws ec2 create-capacity-reservation --region "$REGION" \
        --instance-type "$TYPE" --instance-platform "Linux/UNIX" \
        --availability-zone "$az" --tenancy default --instance-count 1 \
        --instance-match-criteria open --end-date-type unlimited \
        --tag-specifications "ResourceType=capacity-reservation,Tags=[{Key=Name,Value=freyr-llm-qwen}]" \
        --query 'CapacityReservation.CapacityReservationId' --output text 2>&1); then
      state=$(wait_state "$out")
      if [ "$state" = failed ]; then
        log "$out in $az was accepted but then failed (no capacity) - nothing is billed, it disappears after 60 min"
        continue
      fi
      if [ "$state" = active ]; then
        log "RESERVED $out : $TYPE in $az  (active - billing has started)"
      else
        log "REQUESTED $out : $TYPE in $az is still 'pending' after 5 min - check it in the EC2 console (Capacity Reservations)"
      fi
      log "NEXT: launch the stack with InstanceType=$TYPE, GpuAvailabilityZone=$az, GpuHostPlacement=private-a"
      log "      (SecondAvailabilityZone = any other zone, used only for the ALB's second subnet)"
      log "No end date (Manually): it bills until you cancel it."
      log "CANCEL LATER: aws ec2 cancel-capacity-reservation --region $REGION --capacity-reservation-id $out"
      if [ -n "$SNS_TOPIC_ARN" ]; then
        aws sns publish --region "$REGION" --topic-arn "$SNS_TOPIC_ARN" \
          --subject "GPU capacity reserved ($TYPE)" \
          --message "Reserved $out ($TYPE in $az, $state). Launch the stack with GpuAvailabilityZone=$az, GpuHostPlacement=private-a." >/dev/null 2>&1 || true
      fi
      exit 0
    fi
    case "$out" in
      *InsufficientInstanceCapacity*|*"Insufficient capacity"*|*InsufficientCapacity*)
        ;;                                                     # expected: keep trying
      *Unsupported*|*"not supported"*)
        log "$az does not support $TYPE reservations right now (Unsupported) - dropping it from the list"
        rest=(); for z in "${AZS[@]}"; do [ "$z" != "$az" ] && rest+=("$z"); done; AZS=("${rest[@]}")
        if [ "${#AZS[@]}" -eq 0 ]; then log "STOPPING: no zones left to try"; exit 1; fi ;;
      *RequestLimitExceeded*|*Throttling*)
        log "throttled by the EC2 API - backing off 120s"; sleep 120 ;;
      *)
        log "STOPPING on unexpected error in $az: $out"; exit 1 ;;   # quota, permissions, expired session...
    esac
  done
  log "round $round: no $TYPE capacity in ${AZS[*]} - retrying in ${INTERVAL}s"
  sleep "$INTERVAL"
done
