#!/usr/bin/env bash
# verify-llm-host.sh - run on the GPU instance (Session Manager): sudo bash verify-llm-host.sh
# Read-only health check of the bootstrap: GPU, disks, model, containers, vLLM, Bifrost, agents.
# shellcheck disable=SC2015  # ok/bad/info always return 0, so "A && ok || bad" is a safe if/else
[ -n "${BASH_VERSION:-}" ] || { echo "Run with bash: sudo bash $0"; exit 1; }
set -u
ok()   { printf '  \033[32mPASS\033[0m  %s\n' "$*"; }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$*"; }
info() { printf '  ....  %s\n' "$*"; }
hdr()  { printf '\n== %s\n' "$*"; }

hdr "1. Bootstrap (user data)"
if command -v cloud-init >/dev/null; then info "cloud-init: $(cloud-init status 2>/dev/null | tr '\n' ' ')"; fi
if [ -f /var/log/llm-bootstrap.log ]; then
  info "last bootstrap log lines:"; tail -n 5 /var/log/llm-bootstrap.log | sed 's/^/        /'
else bad "/var/log/llm-bootstrap.log missing - user data has not started"; fi

hdr "2. Instance"
T=$(curl -sS -X PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 60')
md() { curl -sS -H "X-aws-ec2-metadata-token: $T" "http://169.254.169.254/latest/meta-data/$1"; }
info "type $(md instance-type), AZ $(md placement/availability-zone), id $(md instance-id), private IP $(md local-ipv4)"
if md public-ipv4 2>/dev/null | grep -qE '^[0-9]'; then bad "instance has a public IP"; else ok "no public IP"; fi

hdr "3. GPU"
if nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader >/tmp/gpu 2>&1; then
  ok "GPU: $(cat /tmp/gpu)"
  info "GPU memory used: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader) (vLLM reserves ~90% once loaded)"
else bad "nvidia-smi failed: $(cat /tmp/gpu)"; fi
if docker info 2>/dev/null | grep -qi nvidia; then ok "Docker has the NVIDIA runtime"; else bad "NVIDIA runtime not registered with Docker"; fi

hdr "4. Disks"
if findmnt /opt/models >/dev/null; then ok "data volume mounted: $(df -h /opt/models | awk 'NR==2{print $2" size, "$5" used"}')"; else bad "/opt/models not mounted"; fi
grep -q '/opt/models' /etc/fstab && ok "/opt/models in /etc/fstab (survives reboot)" || bad "/opt/models not in /etc/fstab"
[ -d /opt/dlami/nvme ] && info "NVMe scratch: $(df -h /opt/dlami/nvme | awk 'NR==2{print $2}') at /opt/dlami/nvme"

hdr "5. Model weights"
M=$(find /opt/models -maxdepth 2 -name config.json -path '*Qwen*' 2>/dev/null | head -1)
if [ -n "$M" ]; then
  d=$(dirname "$M"); ok "model at $d ($(du -sh "$d" | cut -f1), $(find "$d" -name '*.safetensors' | wc -l) safetensors files)"
else bad "model not on /opt/models yet (still downloading?) - see bootstrap log"; fi

hdr "6. Services"
systemctl is-enabled llm-stack >/dev/null 2>&1 && ok "llm-stack.service enabled (restarts on reboot)" || bad "llm-stack.service not enabled"
for c in vllm bifrost; do
  s=$(docker inspect -f '{{.State.Status}} (restarts: {{.RestartCount}})' "$c" 2>/dev/null)
  if [[ "$s" == running* ]]; then ok "container $c: $s"; else bad "container $c: ${s:-not created}"; fi
done
[ -s /run/llm/bifrost.env ] && ok "Bifrost secrets file present ($(stat -c '%a %U' /run/llm/bifrost.env))" || bad "/run/llm/bifrost.env missing"

hdr "7. vLLM (model server, localhost:8000)"
if [ "$(curl -s -o /dev/null -w '%{http_code}' localhost:8000/health)" = 200 ]; then
  ok "vLLM /health = 200"
  info "served models: $(curl -s localhost:8000/v1/models | python3 -c 'import sys,json;print([m["id"] for m in json.load(sys.stdin)["data"]])' 2>/dev/null)"
  docker logs vllm 2>&1 | grep -iE 'KV cache|maximum concurrency' | tail -2 | sed 's/^/        /'
  r=$(curl -s -m 120 localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Reply with exactly: OK"}],"max_tokens":2048,"reasoning_effort":"low"}')
  if echo "$r" | grep -q '"choices"'; then
    summary=$(echo "$r" | python3 -c 'import sys,json
c=json.load(sys.stdin)["choices"][0]; m=c["message"]; th=m.get("reasoning") or m.get("reasoning_content") or ""
print("answer=%r, finish=%s, thinking=%d chars" % ((m.get("content") or "").strip()[:80], c.get("finish_reason"), len(th)))' 2>/dev/null)
    if echo "$summary" | grep -q "answer=''"; then bad "test completion returned no answer: $summary"; else ok "test completion: $summary"; fi
  else bad "test completion failed: ${r:0:300}"; fi
else
  bad "vLLM not healthy yet - first load takes several minutes. Last log lines:"
  docker logs --tail 15 vllm 2>&1 | sed 's/^/        /'
fi

hdr "8. Bifrost (gateway, localhost:8080)"
if [ "$(curl -s -o /dev/null -w '%{http_code}' localhost:8080/health)" = 200 ]; then
  ok "Bifrost /health = 200"
  c=$(curl -s -o /dev/null -w '%{http_code}' localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{"model":"vllm/qwen3.8-27b","messages":[{"role":"user","content":"hi"}],"max_tokens":8}')
  if [ "$c" = 401 ] || [ "$c" = 403 ]; then ok "request without a virtual key rejected ($c)"; else bad "request without a virtual key returned $c (expected 401/403)"; fi
  a=$(curl -s -o /dev/null -w '%{http_code}' localhost:8080/api/governance/virtual-keys)
  if [ "$a" = 401 ]; then ok "admin API locked (no password -> 401)"; else bad "admin API answered $a without a password - lock it (see llm-admin.py / runbook)"; fi
else bad "Bifrost not healthy"; docker logs --tail 15 bifrost 2>&1 | sed 's/^/        /'; fi

hdr "9. Outbound and AWS access"
[ "$(curl -s -o /dev/null -m 10 -w '%{http_code}' https://huggingface.co)" != 000 ] && ok "internet egress via NAT works" || bad "no internet egress (NAT/route?)"
aws sts get-caller-identity --query Arn --output text >/tmp/arn 2>&1 && ok "instance role: $(cat /tmp/arn)" || bad "instance role not usable: $(cat /tmp/arn)"
/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl -a status 2>/dev/null | grep -q '"status": "running"' \
  && ok "CloudWatch agent running" || bad "CloudWatch agent not running"

hdr "10. CloudFormation signal"
info "the bootstrap signals CloudFormation as soon as vLLM and Bifrost are healthy (sections 7-8 PASS)."
info "confirm in the console: stack status CREATE_COMPLETE, GpuHost resource CREATE_COMPLETE."
echo
