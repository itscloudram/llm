# Changelog

## Guide v4: 20 Oct 2026

- **New in section 2:** LLM parameters and core concepts:
  - what "27B" means and where the parameters are;
  - how a transformer writes a token;
  - the architecture in numbers, and parameters to gigabytes;
  - how a model is trained;
  - dense versus mixture-of-experts;
  - a full table of request parameters with Qwen's values;
  - turning thinking off;
  - hallucination;
  - prompting, RAG, fine-tuning and agents.
- **New chapters 18 to 25:**
  - GPU and instance primer;
  - observability and benchmarking;
  - scaling, availability and recovery;
  - cost engineering;
  - model lifecycle;
  - security for LLM applications (OWASP Top 10 for LLM apps);
  - API request reference;
  - template reference: all 44 parameters, 20 outputs and 64 resources, generated from the YAML.
- **Glossary** grown to 57 terms; new sources added.
- **Word file** rebuilt as a navigable document. It has:
  - a cover, quick navigation and a contents page;
  - "at a glance" boxes and colour-coded callouts;
  - 14 charts and diagrams;
  - tick boxes and clickable cross-references;
  - a quick-reference card.
- **Interactive page:** all 27 chapters, plus a sampling playground and cost per million tokens in the calculator.

## v3: 19 Oct 2026

- **Repo hygiene:** example values replace the PoC domain: `llm.example.com` in the parameter files, an empty default endpoint in Qwen Chat. Set `DomainName` to your own.
- **README:** new section on the optional Qwen Chat app.
- **Docs:** the solution guide is now in the repository as an interactive page (`docs/site/`), Markdown (`docs/GUIDE.md`) and Word (`docs/Solution-Guide.docx`). The diagram's IAM label now reads *scoped*, matching the security section.
- **Template:** added `p5.4xlarge` (1x H100 80 GB, bookable as an EC2 Capacity Block) with an automatic 128 parallel requests.
- **Template:** new `MtpSpeculativeTokens` parameter, 0 to 3, default 0 (off). It adds vLLM `--speculative-config` with the model's own MTP layer.
- **Parameters:** new `params-p5.json`.
- **`reserve-gpu.sh`:**
  - skips zones that don't offer the instance type;
  - drops zones that return `Unsupported`;
  - reports success only when the reservation is **active**, since AWS can accept a request and then fail it.
- **`llm-admin.py`:** handles Bifrost returning `virtual_keys: null` when no keys exist.
- **Docs:** quick start uses `--template-url`, because the template is over the 51,200-byte `--template-body` limit.

## v2: 8 to 9 Oct 2026 (`llm-qwen-gpu-bifrost.yaml`)

One template for g6e and g7e, with every fix found on the first live launch:

- **Bifrost `config_store`:** added `"config": {"path": "./config.db"}`. Without it Bifrost crash-looped with *failed to unmarshal sqlite config*.
- **`MaxNumSeqs`:** 0 now means automatic, 64 on g6e and 128 on g7e. vLLM's default of 256 exceeded the 165 per-request state blocks that fit on 48 GB.
- **Bifrost vLLM provider:** set `allow_private_network: true`. Bifrost's SSRF guard returned 502 *connection to private IP … is not allowed*.
- **Admin account:** created at first boot from a generated Secrets Manager password (`governance.auth_config`), so the admin API is never open.
- **New outputs:** `AdminPasswordCommand`, `BifrostAdminUser` and `MaxNumSeqsInUse`.

## v1: 6 to 8 Oct 2026 (`previous/llm-qwen-g7e-bifrost.v1.yaml`)

- **First template, built for g7e.2xlarge.** It worked for the network, edge and bootstrap, but needed three manual fixes on the instance (above). Superseded by v2.
