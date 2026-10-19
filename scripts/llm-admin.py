#!/usr/bin/env python3
"""llm-admin.py - manage Bifrost virtual keys on the GPU host (run in Session Manager).

  sudo python3 llm-admin.py create <app-name> [<app-name> ...]   create one key per app, test it, print it
  sudo python3 llm-admin.py list                                  list keys (id, name, active)
  sudo python3 llm-admin.py revoke <key-id>                       delete a key
  sudo python3 llm-admin.py test <sk-bf-key>                      send one test question through Bifrost
  sudo python3 llm-admin.py status                                health, admin lock, provider settings

Admin credentials: read from /run/llm/bifrost.env (BIFROST_ADMIN_PASSWORD, written at boot from Secrets
Manager) and the username from Bifrost's config.json. If the password is not there (older stacks), you are
asked for it. Nothing secret is passed on a command line.
"""
import base64
import functools
import getpass
import json
import sys
import urllib.error
import urllib.request

BIFROST = "http://localhost:8080"
VLLM = "http://localhost:8000"
ENV_FILE = "/run/llm/bifrost.env"
CONFIG = "/opt/models/bifrost/config.json"


@functools.lru_cache(maxsize=1)
def admin_auth():
    user, pw = "freyr-admin", ""
    try:
        with open(CONFIG) as f:
            cfg = json.load(f)
        user = cfg.get("governance", {}).get("auth_config", {}).get("admin_username", user)
    except (OSError, ValueError):
        pass
    try:
        with open(ENV_FILE) as f:
            for line in f:
                if line.startswith("BIFROST_ADMIN_PASSWORD="):
                    pw = line.split("=", 1)[1].rstrip("\n")
    except PermissionError:
        sys.exit("Run with sudo: sudo python3 llm-admin.py ...")
    except OSError:
        pass
    if not pw:
        pw = getpass.getpass(f"Bifrost admin password for {user}: ")
    return "Basic " + base64.b64encode(f"{user}:{pw}".encode()).decode()


def call(method, url, body=None, headers=None, timeout=30):
    h = {"Content-Type": "application/json", **(headers or {})}
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode()
            return r.status, (json.loads(raw) if raw.strip() else {})
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        try:
            return e.code, json.loads(raw)
        except ValueError:
            return e.code, {"raw": raw[:300]}
    except urllib.error.URLError as e:
        return 0, {"error": str(e.reason)}


def served_model():
    code, d = call("GET", f"{VLLM}/v1/models")
    if code != 200 or not d.get("data"):
        sys.exit(f"vLLM is not ready on {VLLM} ({code}). Wait for it, or check: sudo docker logs --tail 30 vllm")
    return d["data"][0]["id"]


def ask(key, model, question="Reply with exactly: OK"):
    body = {"model": f"vllm/{model}", "messages": [{"role": "user", "content": question}],
            "max_tokens": 1024, "reasoning_effort": "low"}
    code, d = call("POST", f"{BIFROST}/v1/chat/completions", body, {"x-bf-vk": key}, timeout=180)
    if code == 200:
        return True, (d["choices"][0]["message"].get("content") or "").strip()
    return False, f"HTTP {code}: {json.dumps(d)[:300]}"


def key_value(obj):
    # the create response wraps the key; find the sk-bf-... value wherever it is
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "value" and isinstance(v, str) and v.startswith("sk-bf-"):
                return v
            found = key_value(v)
            if found:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = key_value(v)
            if found:
                return found
    return None


def keys_list(auth):
    code, d = call("GET", f"{BIFROST}/api/governance/virtual-keys", headers={"Authorization": auth})
    if code != 200:
        sys.exit(f"Could not list keys: HTTP {code} {json.dumps(d)[:300]}")
    if isinstance(d, list):
        return d
    return d.get("virtual_keys") or []  # Bifrost sends null when there are no keys yet


def cmd_create(names):
    auth, model = admin_auth(), served_model()
    for name in names:
        body = {"name": name, "is_active": True,
                "provider_configs": [{"provider": "vllm", "weight": 1.0, "allowed_models": [model], "key_ids": ["*"]}]}
        code, d = call("POST", f"{BIFROST}/api/governance/virtual-keys", body, {"Authorization": auth})
        if code not in (200, 201):
            print(f"FAIL  {name}: HTTP {code} {json.dumps(d)[:300]}")
            continue
        key = key_value(d)
        if not key:
            print(f"WARN  {name}: created, but no key value in the response - run 'list' or check the dashboard")
            continue
        ok, ans = ask(key, model)
        print(f"{'PASS' if ok else 'FAIL'}  {name}: test answer {ans!r}")
        print(f"      key: {key}   (shown once - store it in the app's secret store)")


def cmd_list():
    for k in keys_list(admin_auth()):
        print(f"{k.get('id', '?'):40}  {k.get('name', '?'):24}  active={k.get('is_active')}")


def cmd_revoke(key_id):
    code, d = call("DELETE", f"{BIFROST}/api/governance/virtual-keys/{key_id}", headers={"Authorization": admin_auth()})
    print("revoked" if code in (200, 204) else f"FAIL HTTP {code} {json.dumps(d)[:300]}")


def cmd_test(key):
    ok, ans = ask(key, served_model(), "Say hello in one short sentence.")
    print(("PASS  " if ok else "FAIL  ") + ans)


def cmd_status():
    model = served_model()
    print(f"PASS  vLLM serving {model}")
    code, _ = call("GET", f"{BIFROST}/health")
    print(f"{'PASS' if code == 200 else 'FAIL'}  Bifrost /health {code}")
    code, _ = call("GET", f"{BIFROST}/api/governance/virtual-keys")
    print(f"{'PASS' if code == 401 else 'FAIL'}  admin API without password -> {code} (expect 401)")
    code, _ = call("POST", f"{BIFROST}/v1/chat/completions",
                   {"model": f"vllm/{model}", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8})
    print(f"{'PASS' if code == 401 else 'FAIL'}  model call without key -> {code} (expect 401)")
    code, p = call("GET", f"{BIFROST}/api/providers/vllm", headers={"Authorization": admin_auth()})
    apn = p.get("network_config", {}).get("allow_private_network") if code == 200 else None
    print(f"{'PASS' if apn else 'FAIL'}  provider vllm allow_private_network = {apn}")
    print(f"....  virtual keys: {len(keys_list(admin_auth()))}")


def main():
    a = sys.argv[1:]
    if not a or a[0] in ("-h", "--help"):
        print(__doc__)
        return
    if a[0] == "create" and len(a) > 1:
        cmd_create(a[1:])
    elif a[0] == "list":
        cmd_list()
    elif a[0] == "revoke" and len(a) == 2:
        cmd_revoke(a[1])
    elif a[0] == "test" and len(a) == 2:
        cmd_test(a[1])
    elif a[0] == "status":
        cmd_status()
    else:
        print(__doc__)
        sys.exit(2)


if __name__ == "__main__":
    main()
