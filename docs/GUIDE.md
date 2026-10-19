# Self-Hosting Qwen3.8-27B on Amazon EC2 with vLLM and Bifrost: Solution Guide

Last updated 19 October 2026 · Author: Ram (Freyr) · **[Interactive version](https://claude.ai/artifact/JmfPaYGYuj8Hgm8hBrgcfA)** (or open [`docs/site/index.html`](site/index.html) from a clone) · **[Word version](Solution-Guide.docx)**

**Contents**

- [1. How to use this guide](#1-how-to-use-this-guide)
- [2. LLM fundamentals every AI infrastructure engineer needs](#2-llm-fundamentals-every-ai-infrastructure-engineer-needs)
- [3. Why self-host an open-weight model](#3-why-self-host-an-open-weight-model)
- [4. Qwen3.8-27B: the facts this build relies on](#4-qwen38-27b-the-facts-this-build-relies-on)
- [5. vLLM: the model server](#5-vllm-the-model-server)
- [6. The LLM gateway: why Bifrost sits in front of vLLM](#6-the-llm-gateway-why-bifrost-sits-in-front-of-vllm)
- [7. Architecture](#7-architecture)
- [8. Choosing the GPU](#8-choosing-the-gpu)
- [9. Repository and files](#9-repository-and-files)
- [10. Prerequisites](#10-prerequisites)
- [11. Step-by-step deployment](#11-step-by-step-deployment)
- [12. Inside the EC2 instance: what the bootstrap installs and why](#12-inside-the-ec2-instance-what-the-bootstrap-installs-and-why)
- [13. After deployment: verify, create keys, make the first call](#13-after-deployment-verify-create-keys-make-the-first-call)
- [14. Operations](#14-operations)
- [15. Security model](#15-security-model)
- [16. Troubleshooting and lessons learned](#16-troubleshooting-and-lessons-learned)
- [17. Teardown](#17-teardown)
- [18. Glossary](#18-glossary)
- [19. References](#19-references)

## 1. How to use this guide

This guide takes you from an empty AWS account to a private, OpenAI-compatible Qwen3.8-27B endpoint that agents and applications call over HTTPS. One CloudFormation template builds everything, and every step was run and verified on a real instance in us-east-1 in October 2026.

**What you end up with**

- `https://<your-domain>/v1/chat/completions`: an OpenAI-compatible API in front of a model you host yourself.
- Every caller needs its own **virtual key**, so you can see, limit or revoke each application separately.
- The GPU instance has no public IP. Only a load balancer with a web firewall (WAF) faces the internet.
- Streaming answers, visible reasoning, tool calling and image input, served by vLLM on one NVIDIA GPU.

**Pick your path**

| You are | Read in this order | Time to a working endpoint |
| --- | --- | --- |
| New to AI infrastructure | Sections 2 to 7 (concepts), then 10 to 13 (build), then 16 (problems seen) | A day, including reading |
| Comfortable with AWS, new to LLM serving | Sections 2, 5, 6 and 8, then 10 to 13 | Half a day |
| Experienced, need to deploy | Sections 9 to 13, keep 16 open | About an hour, mostly waiting for the model to download and load |

**Conventions**

- `Monospace` is something you type or a value you will see. Commands run in **CloudShell** (AWS console, top bar) unless marked **SSM**, which means a Session Manager shell on the GPU instance.
- Every fact comes with its source: a link to official documentation, or **Measured** when it was observed on our own instance.
- Prices are On-Demand, us-east-1, Linux, as checked in October 2026. Check them again before you budget.
- Region is us-east-1 throughout. The template enforces it, and section 8 explains how to change it.

## 2. LLM fundamentals every AI infrastructure engineer needs

An LLM server spends its time in two very different phases. Reading the prompt (**prefill**) is limited by GPU compute. Writing the answer one token at a time (**decode**) is limited by how fast the GPU can read its own memory. Almost every sizing decision in this guide follows from that split.

### Tokens and the chat template

Models read and write **tokens**, pieces of words, not characters. Your message is wrapped in a **chat template** (role markers, the start of the assistant's turn, the thinking tag) before the model sees it. **Measured:** the one-word message `Hi` cost 41 input tokens, because the template adds about 40 fixed tokens to every request.

### Context window

The context window is the most tokens one request can hold: prompt, chat history, attached files, thinking and answer combined. Qwen3.8-27B supports 262,144 tokens natively ([config](https://huggingface.co/Qwen/Qwen3.8-27B-FP8/raw/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a/config.json): `max_position_embeddings: 262144`). On the 48 GB g6e this guide caps it at 65,536 (`MaxModelLen`) to leave GPU memory for the cache described next. A request over the cap is rejected with HTTP 400, not silently trimmed.

### Prefill, decode and why one request runs at about 25 tokens/s

To write each new token, the GPU reads every model weight from its memory once. So the most tokens per second one request can get is roughly memory bandwidth divided by model size:

```math
\text{tokens/s}_{\max} \approx \frac{\text{GPU memory bandwidth}}{\text{bytes of weights}} = \frac{864\ \text{GB/s (L40S)}}{31\ \text{GB}\ (28.9\ \text{GiB})} \approx 28
```

**Measured** on g6e.2xlarge: 25.0 tokens/s, about 90% of that ceiling. Running more requests at once raises total throughput, because each memory read serves every request in the batch, but each user still sees about 25 tokens/s. Faster memory is the only hardware fix; section 8 compares the options.

Prefill is different: the whole prompt is processed in parallel, so it is limited by compute and shows up as **time to first token**. **Measured:** 401 ms for a short prompt, including the round trip from Bangalore to us-east-1.

**Evidence.** NVIDIA: *"Model execution is frequently memory-bandwidth bound—in particular, bandwidth-bound in the weights"* ([NVIDIA technical blog](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)). Databricks: *"Generating the first token is typically compute-bound, while subsequent decoding is memory-bound"* ([Databricks engineering blog](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices)). The L40S's 864 GB/s comes from [NVIDIA's L40S page](https://www.nvidia.com/en-us/data-center/l40s/).

### KV cache: the memory that limits context and concurrency

While generating, the model keeps a key and a value vector for every past token in every attention layer: the **KV cache**. Its size per token follows from the model config:

```math
\text{KV bytes per token} = 2 \times \text{attention layers} \times \text{KV heads} \times \text{head dim} \times \text{bytes} = 2 \times 16 \times 4 \times 256 \times 2 = 65{,}536
```

That is 64 KiB per token, or 4 GiB for one full 65,536-token request. It is small for a 27B model because Qwen3.8 is a **hybrid**: only 16 of its 64 layers use full attention, and the other 48 use linear attention (Gated DeltaNet), which keeps a fixed-size state per request instead of a growing cache. That fixed state is why vLLM reported only 165 request slots on the L40S and refused its default of 256 parallel requests (section 16).

### Quantization: BF16 versus FP8

The original weights are BF16 (2 bytes per parameter, about 54 GB), which doesn't fit a 48 GB GPU. Qwen publishes an official **FP8** checkpoint (1 byte per parameter, *"fine-grained fp8 quantization with block size of 128"*), about 29 GB, with near-identical quality ([FP8 model card](https://huggingface.co/Qwen/Qwen3.8-27B-FP8)). vLLM documents FP8 computation on GPUs with compute capability 8.9 or higher (Ada, Hopper, Blackwell), and a weight-only fallback through Marlin kernels elsewhere ([vLLM FP8 docs](https://docs.vllm.ai/en/latest/features/quantization/llm_compressor/fp8/)). **Measured:** on the L40S, an Ada GPU, vLLM still chose *"Weight-only FP8 compression … leveraging the Marlin kernel"* for this block-scaled checkpoint. The most likely reason is that vLLM's kernels for this block-scaled format don't cover Ada; I haven't confirmed that in vLLM's source. Either way, weights stay FP8 in memory while maths runs in 16-bit. That slows prefill but barely affects decode, which is limited by memory. The H100 and RTX PRO 6000 run this checkpoint with FP8 maths.

### Reasoning (thinking)

Qwen3.8 can think before it answers. vLLM's reasoning parser splits that text into a separate `reasoning` field, so applications can show or hide it. Two things matter in practice:

- Thinking tokens count toward `max_tokens`. Set it too low and the model can run out while still thinking and return an empty answer with `finish_reason: length`.
- `reasoning_effort` accepts `low`, `medium` or `xhigh` for this model. `high` returns HTTP 500 because the chat template rejects it ([Qwen3.8 issue #217](https://github.com/QwenLM/Qwen3.8/issues/217)).
- Through Bifrost, `xhigh` can fail too: Bifrost's OpenAI converter rewrites `xhigh` to `high` for vLLM ([Bifrost issue #6193](https://github.com/maximhq/bifrost/issues/6193), fix proposed, not confirmed released). Use `low` or `medium` through the gateway until you've tested `xhigh` on your Bifrost version.

### Tool calling

Agents give the model a list of tools as JSON schemas. The model writes its tool calls in its own XML format, and vLLM's `qwen3_xml` tool-call parser converts them into standard OpenAI `tool_calls`. Agent frameworks therefore work unchanged.

### Sampling

`temperature` and `top_p` control how adventurous each next token is. The model card recommends `temperature 1.0`, `top_p 0.95`, `top_k 20` in thinking mode, and `temperature 0.7`, `top_p 0.8` in non-thinking mode ([model card](https://huggingface.co/Qwen/Qwen3.8-27B-FP8)).

### Multi-token prediction (MTP)

The model ships with one extra prediction layer (`mtp_num_hidden_layers: 1`) that drafts several tokens ahead. The full model then checks those drafts in a single pass. The vLLM recipe enables it with `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'` ([vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B)). It's the cheapest speed-up available and is off in this build until it's measured (section 14).

### Multimodal input

The checkpoint includes a vision encoder (`vision_config` in the config). **Measured:** vLLM loaded it at start-up (*"Encoder cache will be initialized … profiled with 1 image items"*). Images are sent as OpenAI `image_url` content parts.

## 3. Why self-host an open-weight model

Self-hosting buys control: data never leaves your account, the model never changes under you, and you can tune it. You pay for that in GPU hours and operating effort, and it's only cheaper than an API when the GPU is kept busy.

| Consideration | Hosted API (closed or open model) | Self-hosted open weights (this build) |
| --- | --- | --- |
| Where prompts and answers go | To the provider, under their terms | Stay in your VPC; nothing is logged by Bifrost (`disable_content_logging`) |
| Model changes | Provider can update or retire it | Pinned to one commit until you choose to move |
| Customisation | Limited to what the provider exposes | Fine-tuning, LoRA adapters, custom chat templates, quantization choices |
| Cost shape | Per token, nothing when idle | Per GPU hour, whether busy or idle |
| Capacity risk | Provider's rate limits | GPU availability (section 16: g7e could not be obtained) |
| Operating effort | None | You patch, monitor and upgrade it |
| Licence | Provider terms | Apache 2.0 for Qwen3.8: commercial use allowed |

**What the industry says.**

- a16z's 2024 enterprise survey found the top reason to adopt open models was *"Control (security of proprietary data and understanding why models produce certain outputs)"*, and that control and customisation *"far outweighed cost as the primary reasons to adopt open source"* ([a16z](https://a16z.com/generative-ai-enterprise-2024/)).
- Menlo Ventures' mid-2025 update lists *"greater customization, potential cost savings"* and *"the ability to deploy within private cloud or on-premises environments"* as the advantages. It also reports that only 13% of AI workloads used open models, down from 19% six months earlier ([Menlo Ventures](https://menlovc.com/perspective/2025-mid-year-llm-market-update/)). Self-hosting is a deliberate choice for control, not the default.

**The cost maths, honestly.**

- **One user at a time:** g6e.2xlarge costs $2.242 per hour. At the measured 25 tokens/s for a single request, a fully busy single stream writes about 90,000 tokens an hour. That's roughly $25 per million output tokens, which is more than many hosted APIs charge.
- **Many users at once:** vLLM batches up to 64 requests, and each GPU memory read serves all of them, so total throughput rises well above 25 tokens/s. We haven't measured that ceiling yet. Run `vllm bench serve` against the instance before using cost as the argument for self-hosting.
- **Idle hours:** stop the instance when nobody uses it (section 14). An idle GPU costs the same as a busy one.

## 4. Qwen3.8-27B: the facts this build relies on

Qwen3.8-27B is an Apache-2.0, 27-billion-parameter vision-language model from Alibaba's Qwen team, released in August 2026. It is a hybrid of linear and full attention, which keeps its memory needs low for its size. This table is the single source of truth for every model value used in the template.

| Property | Value | Source |
| --- | --- | --- |
| Licence | Apache 2.0: commercial use, modification and self-hosting allowed | [Model card](https://huggingface.co/Qwen/Qwen3.8-27B) |
| Parameters | 27B (Hugging Face metadata rounds to 28B) | [Model card](https://huggingface.co/Qwen/Qwen3.8-27B) |
| Type | Causal language model with vision encoder; understands images and video | [Model card](https://huggingface.co/Qwen/Qwen3.8-27B) |
| Layers | 64: 16 × (3 × Gated DeltaNet + 1 × Gated Attention), each followed by an FFN | [Model card](https://huggingface.co/Qwen/Qwen3.8-27B) |
| Full-attention layers | 16, with 24 query heads, 4 KV heads, head dimension 256 | [config.json](https://huggingface.co/Qwen/Qwen3.8-27B-FP8/raw/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a/config.json) |
| Context | 262,144 tokens natively, extensible to about 1,000,000 with YaRN | [Model card](https://huggingface.co/Qwen/Qwen3.8-27B) |
| Thinking | On by default; `reasoning_effort` = `xhigh` (default), `medium`, `low`; can be turned off per request | [Model card](https://huggingface.co/Qwen/Qwen3.8-27B) |
| Multi-token prediction | One MTP layer, trained with multiple steps | [Model card](https://huggingface.co/Qwen/Qwen3.8-27B), config `mtp_num_hidden_layers: 1` |
| Checkpoint served | `Qwen/Qwen3.8-27B-FP8`, pinned to commit `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` | [config.json at that commit](https://huggingface.co/Qwen/Qwen3.8-27B-FP8/raw/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a/config.json) |
| FP8 format | `quant_method: fp8`, `fmt: e4m3`, `weight_block_size: [128, 128]` | [config.json](https://huggingface.co/Qwen/Qwen3.8-27B-FP8/raw/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a/config.json) |
| Size on disk | 29 GB in 66 safetensors files | Measured (`verify-llm-host.sh`) |
| GPU memory for weights | 28.9 GiB | Measured (vLLM log: *"Model loading took 28.9 GiB memory"*) |

**Published benchmark scores** (from the model card, reported by the Qwen team and not re-run here): GPQA Diamond 89.2, LiveCodeBench v6 90.3, SWE-bench Pro 61.7, Terminal Bench 2.1 73.0, OSWorld-Verified 84.3.

**Why the template pins a commit.** Hugging Face repositories can change. Pinning the 40-character commit means every rebuild downloads exactly the weights this guide was tested with, and the S3 copy is stored under that commit's path.

## 5. vLLM: the model server

vLLM turns model weights into a fast, OpenAI-compatible HTTP service. It's the most widely adopted open-source LLM inference engine: AWS, Google Cloud, Red Hat and NVIDIA all build on it or document it.

### What an inference server does

Loading weights into a GPU is the easy part. The server's real job is to keep the GPU busy while many requests of different lengths arrive and finish at different times, without running out of memory. vLLM does that with two ideas:

- **PagedAttention.** The KV cache is stored in fixed-size blocks, like pages of operating-system memory, instead of one contiguous slab per request. The vLLM paper says KV memory *"can be significantly wasted by fragmentation and redundant duplication"* when managed naively, and reports *"2-4×"* higher throughput than FasterTransformer and Orca ([arXiv 2309.06180](https://arxiv.org/abs/2309.06180)). The launch post puts the waste of earlier systems at *"60% – 80%"* versus *"under 4%"* ([vLLM blog](https://vllm.ai/blog/2023-06-20-vllm)).
- **Continuous batching.** A new request joins the running batch as soon as another finishes, instead of waiting for the whole batch. The Orca paper introduced this *"iteration-level"* scheduling ([OSDI '22](https://www.usenix.org/conference/osdi22/presentation/yu)), and Anyscale measured *"23x throughput"* gains from it ([Anyscale](https://www.anyscale.com/blog/continuous-batching-llm-inference)).

### What this build uses

| vLLM feature | Flag or behaviour here | Why |
| --- | --- | --- |
| OpenAI-compatible API | `/v1/chat/completions`, `/v1/models` | Every SDK and agent framework works unchanged |
| Reasoning parser | `--reasoning-parser qwen3` | Splits thinking into a separate `reasoning` field |
| Tool calling | `--enable-auto-tool-choice --tool-call-parser qwen3_xml` | Turns Qwen's XML tool calls into OpenAI `tool_calls` |
| Memory budget | `--gpu-memory-utilization 0.90` | Leaves 10% of GPU memory for the runtime and the vision encoder |
| Context cap | `--max-model-len 65536` (g6e) / `131072` (g7e, p5) | Fits the KV cache in the memory left after weights |
| Concurrency cap | `--max-num-seqs 64` (g6e) / `128` | Fits the per-request linear-attention state (section 16) |
| FP8 weights | Read from the checkpoint automatically | Halves memory versus BF16 |
| Prefix caching | vLLM's default for most models; not confirmed for this hybrid model (check the start-up log) | Repeated system prompts and chat history skip prefill |
| Speculative decoding (MTP) | `--speculative-config` when `MtpSpeculativeTokens` > 0 | Faster decode for memory-bound workloads |
| Multimodal | Vision encoder loaded automatically | Image input through `image_url` parts |

The [vLLM documentation](https://docs.vllm.ai/en/latest/) lists the full set: *"Continuous batching of incoming requests, chunked prefill, prefix caching"*, *"OpenAI-compatible API server"*, FP8 among many quantization formats, and *"Tool calling and reasoning parsers"*. The [MTP page](https://docs.vllm.ai/en/latest/features/speculative_decoding/mtp/) explains that MTP needs no separate draft model.

### Evidence that vLLM is the industry standard

| Who | What they say | Source |
| --- | --- | --- |
| PyTorch Foundation | *"welcome vLLM as a PyTorch Foundation-hosted project"*, 7 May 2025, under neutral governance | [PyTorch blog](https://pytorch.org/blog/pytorch-foundation-welcomes-vllm/) |
| AWS SageMaker | LMI v15 container *"powered by vLLM 0.8.4 with support for the vLLM V1 engine"* | [AWS ML blog](https://aws.amazon.com/blogs/machine-learning/supercharge-your-llm-performance-with-amazon-sagemaker-large-model-inference-container-v15/) |
| AWS Deep Learning Containers | *"now support vLLM images that are optimized for large language model serving"* | [AWS docs](https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/dlc-vllm-ec2.html) |
| Amazon Rufus | *"a multi-node inference solution using Amazon Trainium and vLLM"* | [AWS ML blog](https://aws.amazon.com/blogs/machine-learning/how-amazon-scaled-rufus-by-building-multi-node-inference-using-aws-trainium-chips-and-vllm) |
| Google Cloud | GKE tutorial: *"Deploy a vLLM container to your cluster"* | [GKE docs](https://docs.cloud.google.com/kubernetes-engine/docs/tutorials/serve-gemma-gpu-vllm) |
| Red Hat | *"Red Hat AI Inference Server, powered by vLLM"* | [Business Wire](https://www.businesswire.com/news/home/20250520450429/en) |
| NVIDIA Dynamo | *"vLLM offers the broadest feature coverage in Dynamo"* | [NVIDIA docs](https://docs.nvidia.com/dynamo/v1.2.0/resources/feature-matrix) |
| Open source | 93.7k GitHub stars, Apache 2.0 licence (19 Oct 2026) | [GitHub](https://github.com/vllm-project/vllm) |

**Alternatives, briefly.**

- **SGLang:** a strong open-source peer; Qwen lists it alongside vLLM.
- **TensorRT-LLM:** NVIDIA-only and needs an engine build step.
- **Hugging Face Transformers:** fine for experiments, not for serving many users.

This build uses vLLM because it has a maintained Qwen3.8 recipe, the parsers this model needs, and the widest deployment record.

## 6. The LLM gateway: why Bifrost sits in front of vLLM

vLLM serves the model but doesn't know who is calling it. A gateway adds identity, limits, accounting and routing in one place, so every application talks to one stable endpoint and each one can be controlled separately.

### What a gateway adds

| Without a gateway | With Bifrost in front |
| --- | --- |
| One shared secret, or none, for every caller | A **virtual key** per application, revocable on its own |
| No idea which app used how much | Usage recorded per key, team or customer |
| Any caller can flood the GPU | Rate limits and budgets per key |
| Changing model or provider means changing every app | Apps call `vllm/qwen3.8-27b`; routing changes happen in the gateway |
| One model server is a single point of failure | Weighted routing and fallbacks across providers or instances (not used in this PoC) |

### This is an established pattern, not a one-off

- **Gartner** published a *Market Guide for AI Gateways* in October 2025: *"AI gateways enable more effective governance of AI applications and agents. They reduce the risk of unexpected costs from AI providers and protect private data in API traffic"* ([Gartner abstract](https://www.gartner.com/en/documents/7051698)).
- **AWS** publishes *Guidance for a Multi-Provider Generative AI Gateway*, *"a unified, industry-standard API gateway based on OpenAI API standards"* built on LiteLLM. It covers budget controls, rate limiting, load balancing and failover, logging, prompt caching and guardrails ([AWS Guidance](https://docs.aws.amazon.com/solutions/multi-provider-generative-ai-gateway-on-aws/), [reference architecture](https://aws.amazon.com/blogs/machine-learning/streamline-ai-operations-with-the-multi-provider-generative-ai-gateway-reference-architecture)).
- **Cloudflare** and **Kong** sell AI gateways with the same core features: rate limits, caching, retries and fallbacks, and PII redaction ([Cloudflare](https://developers.cloudflare.com/ai-gateway/), [Kong](https://developer.konghq.com/ai-gateway/)).

### Bifrost compared with LiteLLM

|  | [Bifrost](https://github.com/maximhq/bifrost) | [LiteLLM](https://github.com/BerriAI/litellm) |
| --- | --- | --- |
| Licence | Apache 2.0 | MIT (except an `enterprise/` folder) |
| Language and footprint | Go, one container, SQLite config store | Python; the proxy usually runs with PostgreSQL |
| Virtual keys, budgets, rate limits | Yes: *"Virtual Keys are the primary governance entity in Bifrost"* ([docs](https://docs.getbifrost.ai/features/governance/virtual-keys)) | Yes: per-key spend tracking, `max_budget`, `tpm_limit` / `rpm_limit` ([docs](https://docs.litellm.ai/docs/proxy/virtual_keys)) |
| vLLM provider | Built in: *"a self-hosted inference provider with OpenAI- and Anthropic-compatible API surfaces"* ([docs](https://docs.getbifrost.ai/providers/supported-providers/vllm)) | Built in |
| Published overhead | 0.99 ms vs 40 ms for LiteLLM at 500 req/s on a t3.medium; 424 vs 44.84 req/s (vendor benchmark, [Maxim](https://www.getmaxim.ai/bifrost/resources/benchmarks)) | (none of its own in that comparison) |
| Adoption, 19 Oct 2026 | 8.7k GitHub stars | 61.3k GitHub stars; basis of the AWS reference architecture |

**Why Bifrost here, honestly.**

- **Low overhead in one container:** that suits a single GPU host where the gateway runs beside vLLM.
- **The benchmark is the vendor's own:** treat its numbers as a claim, not a measurement.
- **LiteLLM is more widely adopted:** it's also the gateway in AWS's reference architecture, so it's the safer choice if a team already runs it.
- **Switching is cheap:** both expose the same OpenAI API, so callers don't change.

**Bifrost behaviours that bit us, all now handled in the template:**

- **It blocks private IPs by default.** That protects against server-side request forgery, but it also blocks vLLM. Fix: `allow_private_network: true` on the vLLM provider only.
- **Its admin API is open until an admin exists.** Fix: the admin account is created at first boot from Secrets Manager.
- **It may rewrite `reasoning_effort: xhigh` to `high` for vLLM.** That's a known bug ([issue #6193](https://github.com/maximhq/bifrost/issues/6193)). Use `low` or `medium` through the gateway.

## 7. Architecture

One private GPU instance runs two containers: Bifrost (the gateway) in front of vLLM (the model server). The only thing reachable from the internet is an Application Load Balancer protected by AWS WAF, and it forwards nothing but `/v1/*` to Bifrost.

![Architecture: callers reach WAF and the ALB in public subnets, which forward /v1/* to Bifrost and vLLM on a private GPU instance](architecture.png)

### The path of one request

```mermaid
sequenceDiagram
    autonumber
    participant App as Caller (agent, app, curl)
    participant DNS as Route 53 (other account)
    participant Edge as AWS WAF + ALB (public subnets)
    participant GW as Bifrost :8080 (private EC2)
    participant LLM as vLLM :8000 + GPU
    App->>DNS: resolve your domain
    DNS-->>App: CNAME to the ALB
    App->>Edge: HTTPS POST /v1/chat/completions + x-bf-vk
    Edge->>Edge: WAF rules, rate limit, TLS ends
    Edge->>GW: forward /v1/* only
    GW->>GW: check virtual key (401 if missing or wrong)
    GW->>LLM: OpenAI request, model qwen3.8-27b
    LLM-->>GW: streamed tokens (reasoning, then content)
    GW-->>Edge: stream + usage
    Edge-->>App: stream to the caller
```

1. **DNS.** The caller resolves your domain. The CNAME lives in the Route 53 account and points at the load balancer.
2. **Edge.** AWS WAF checks the request (managed rule sets, IP reputation, rate limit). The ALB ends TLS with your ACM certificate and forwards only `/v1/*` to port 8080 on the instance.
3. **Gateway.** Bifrost checks the virtual key. No key or a wrong key gets HTTP 401 here, before the GPU does any work. It then applies that key's model access, plus any budget or rate limit you set on the key in the dashboard (none by default), and records the usage.
4. **Model server.** Bifrost forwards to vLLM on port 8000 over a private Docker network. vLLM queues the request into its continuous batch.
5. **GPU.** The model reads the prompt (prefill), then streams tokens back through Bifrost and the ALB to the caller.

### Every component, and whether you need it

| Component | What it does here | Required? | Why it is there |
| --- | --- | --- | --- |
| VPC with 2 public and 2 private subnets | Network for everything | Yes | An ALB needs subnets in two Availability Zones; the GPU host lives in a private one |
| Internet gateway | Lets the ALB and NAT reach the internet | Yes | Without it nothing in the VPC is reachable |
| NAT gateway | Outbound-only internet for the private instance | Yes | apt packages, Docker images and the first Hugging Face download |
| S3 gateway endpoint | Private path from the instance to S3 | Strongly recommended | Weights load from S3 without going through NAT, which avoids NAT data charges |
| Application Load Balancer | TLS, health checks, single public entry | Yes | Keeps the instance private; 600 s idle timeout so long answers don't get cut off |
| AWS WAF | Filters bad traffic before it reaches the ALB | Recommended | Rate limiting and managed rules; body-size rules are set to count only, so long prompts aren't blocked |
| ACM certificate | HTTPS for your domain | Yes | Must be in the same account and Region as the ALB |
| EC2 GPU instance | Runs Bifrost, vLLM and the model | Yes | g6e.2xlarge by default; section 8 compares the alternatives |
| EBS gp3 data volume, 150 GiB | Model copy, vLLM compile cache, Bifrost database | Yes | Survives stop and start, so restarts take about 3 minutes |
| S3 model bucket | The pinned weights, downloaded once from Hugging Face | Recommended | Later launches load from S3 instead of re-downloading |
| Secrets Manager | Bifrost admin password, encryption key, setup token, optional Hugging Face token | Yes | No secret in the template, the user data or the instance image |
| IAM instance role | Read the bucket, the secrets and write logs | Yes | Scoped to this stack's bucket, secrets and logs, plus two AWS managed policies (SSM, CloudWatch agent); no access keys on the instance |
| Session Manager | Shell access and the dashboard tunnel | Yes | No SSH keys and no inbound ports |
| CloudWatch logs, metrics, alarms and SNS | vLLM, Bifrost and bootstrap logs; GPU metrics; alerts by email | Recommended | You can troubleshoot without logging in |
| Data Lifecycle Manager | Daily snapshots of both EBS volumes (root and data) at 20:30 UTC (02:00 IST), keeps 7 | Optional | Fast recovery of the model copy and Bifrost state |
| EventBridge Scheduler | Stops and starts the instance on office hours | Optional (off by default) | Saves GPU cost overnight |
| Clean-up Lambda | Empties the buckets and deletes the stack's snapshots on stack delete | Yes | Without it, a non-empty bucket blocks stack deletion |

## 8. Choosing the GPU

Pick by GPU memory first (does the model plus its cache fit?), then by memory bandwidth (how fast each answer streams), then by whether you can actually get the capacity. For this model, a single H100 on p5.4xlarge is the best performer you can guarantee; g6e.2xlarge is the cheapest that works.

**Sizing rule of thumb.** Weights 28.9 GiB, plus at least one full-length request's KV cache (64 KiB per token, so 4 GiB at 65,536 tokens), plus the per-request linear-attention state, plus about 10% headroom (`gpu-memory-utilization 0.90`). 48 GB works with limits; 80 GB or more is comfortable.

| Instance | GPU | GPU memory | Memory bandwidth | Single-request ceiling¹ | On-Demand, us-east-1 | Getting capacity |
| --- | --- | --- | --- | --- | --- | --- |
| [g6e.2xlarge](https://aws.amazon.com/ec2/instance-types/g6e/) | 1× L40S | 48 GB | 864 GB/s | about 28 tokens/s (**measured 25**) | $2.242/hr | Capacity Reservation; obtained after retries |
| [g7e.2xlarge](https://aws.amazon.com/ec2/instance-types/g7e/) | 1× RTX PRO 6000 Blackwell | 96 GB | 1,597 GB/s | about 52 tokens/s | $3.363/hr | Not obtained in us-east-1 (1a, 1c) during the PoC |
| [p5.4xlarge](https://aws.amazon.com/ec2/instance-types/p5/) | 1× H100 | 80 GB HBM3 | 3,350 GB/s | about 108 tokens/s | [$6.88/hr](https://www.doit.com/compute/compute/aws/us-east-1/p5.4xlarge) (spot about $3.76–4.67) | **Capacity Block**: guaranteed for dates you book |
| [g6e.12xlarge](https://aws.amazon.com/ec2/instance-types/g6e/) | 4× L40S | 192 GB | 4 × 864 GB/s | about 1.8–2.5× g6e.2xlarge² | [$10.49/hr](https://www.devzero.io/instances/aws/g6e.12xlarge) | **Not supported by this template** (single-GPU only); listed for comparison |

¹ Memory bandwidth ÷ 31 GB (28.9 GiB) of weights: the theoretical maximum for one request (section 2). Real results land at about 85–90% of it, as measured on g6e. ² Estimate: the model would be split across 4 GPUs (tensor parallelism), and the L40S has no NVLink, so the GPUs exchange data over PCIe and don't reach 4×. Using it would need a `--tensor-parallel-size 4` flag the template doesn't have. Bandwidth figures: [L40S](https://www.nvidia.com/en-us/data-center/l40s/), [RTX PRO 6000](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/), [H100](https://www.nvidia.com/en-us/data-center/h100/).

### Getting the capacity

- **On-Demand Capacity Reservation** (g6e, g7e): reserves capacity *"in a specific Availability Zone for any duration"* and is *"charged at the equivalent On-Demand rate whether you run instances in reserved capacity or not"* ([AWS docs](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-capacity-reservations.html), [billing](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/capacity-reservations-pricing-billing.html)). It's immediate or nothing: when the zone has no free host, the request fails. That's why `reserve-gpu.sh` retries.
- **EC2 Capacity Blocks for ML** (p5.4xlarge, not g6e or g7e): books GPUs *"on a future date"*, up to 8 weeks ahead. You pay up front when you book, and you can't cancel ([AWS docs](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-capacity-blocks.html), [billing](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/capacity-blocks-pricing-billing.html)). Supported p5.4xlarge Regions include us-east-1, us-east-2, us-west-2 and ap-south-1 (Mumbai), which is closer to Bangalore.
- **Spot** (any type): about 32–45% cheaper for p5.4xlarge, but AWS can reclaim it with two minutes' notice. Good for experiments, not for a demo.

### Recommendation

1. **Now, free:** keep g6e.2xlarge and test multi-token prediction (`MtpSpeculativeTokens=2`; section 14).
2. **For a demo or pilot:** book a p5.4xlarge Capacity Block and launch with `params-p5.json`. Expect roughly 3–4× faster answers per request, native FP8, 128 parallel requests and 131,072-token context.
3. **Another Region:** the template is fixed to us-east-1 in five places: the RegionIsUsEast1 rule, the ELB log-delivery account mapping (a different account ID per Region), the CertificateArn pattern, the Availability Zone defaults, and REGION in reserve-gpu.sh. Change all five to move Region.

## 9. Repository and files

Everything this guide uses lives in the private GitHub repository [itscloudram/llm](https://github.com/itscloudram/llm), branch `main`, first commit `81f8567` (19 October 2026). Grant teammates read access in the repository's settings. The links below only open for people with access.

| File | Runs in | What it's for | Guide section |
| --- | --- | --- | --- |
| [cloudformation/llm-qwen-gpu-bifrost.yaml](https://github.com/itscloudram/llm/blob/main/cloudformation/llm-qwen-gpu-bifrost.yaml) | CloudFormation | **Current template (v3)**: network, edge, GPU host, gateway, secrets, alarms, snapshots, clean delete. g6e, g7e or p5.4xlarge; optional MTP | 7, 11, 12 |
| [cloudformation/params/params-g6e.json](https://github.com/itscloudram/llm/blob/main/cloudformation/params/params-g6e.json) | CloudFormation | g6e.2xlarge: 65,536 context, 64 parallel requests | 11 |
| [cloudformation/params/params-g7e.json](https://github.com/itscloudram/llm/blob/main/cloudformation/params/params-g7e.json) | CloudFormation | g7e.2xlarge: 131,072 context, 128 parallel requests | 11 |
| [cloudformation/params/params-p5.json](https://github.com/itscloudram/llm/blob/main/cloudformation/params/params-p5.json) | CloudFormation | p5.4xlarge: 131,072 context, 128 parallel requests | 8, 11 |
| [cloudformation/previous/llm-qwen-g7e-bifrost.v1.yaml](https://github.com/itscloudram/llm/blob/main/cloudformation/previous/llm-qwen-g7e-bifrost.v1.yaml) and [its parameters](https://github.com/itscloudram/llm/blob/main/cloudformation/previous/llm-qwen-g7e-bifrost.v1.params.json) | Reference only | First template (v1, g7e only). It needed three manual fixes on the instance (section 16). Kept for history; don't deploy it | 16 |
| [scripts/reserve-gpu.sh](https://github.com/itscloudram/llm/blob/main/scripts/reserve-gpu.sh) | CloudShell | Retries an On-Demand Capacity Reservation until one is active | 11 |
| [scripts/verify-llm-host.sh](https://github.com/itscloudram/llm/blob/main/scripts/verify-llm-host.sh) | Instance (SSM) | Read-only check of GPU, disks, model, containers, vLLM, Bifrost, admin lock, egress | 13 |
| [scripts/llm-admin.py](https://github.com/itscloudram/llm/blob/main/scripts/llm-admin.py) | Instance (SSM) | Create, list, revoke and test virtual keys; status summary | 13 |
| [apps/qwen-chat/](https://github.com/itscloudram/llm/tree/main/apps/qwen-chat) | Laptop | Local chat UI: streaming, thinking, files, timing summary (`npm start`) | 13 |
| [docs/architecture.png](https://github.com/itscloudram/llm/blob/main/docs/architecture.png) |  | The architecture diagram in section 7 | 7 |
| [README.md](https://github.com/itscloudram/llm/blob/main/README.md) and [CHANGELOG.md](https://github.com/itscloudram/llm/blob/main/CHANGELOG.md) |  | Quick start, and what changed in v1, v2 and v3 and why | All |

**How the files were checked before publishing:**

- **Template:** cfn-lint 1.57 reports 0 findings, including informational ones, and the startup script passes ShellCheck with MTP on and off.
- **Parameter files:** all three pass every parameter constraint in the template.
- **`reserve-gpu.sh`:** tested against a fake AWS CLI (7 cases); `llm-admin.py` and the chat app were tested against a mock Bifrost.
- **The running PoC:** v2 ran live on g6e.2xlarge, and v3 adds only p5.4xlarge, the MTP switch and a description change.
- **Not yet run on real hardware:** g7e and p5.4xlarge launches, and MTP turned on.

**Keeping the guide and the repository in step:** when you change a file, update the CHANGELOG and the section listed in the last column.

## 10. Prerequisites

You need five things before the first launch. Each one caused a failed or blocked attempt during the PoC when it was missing.

| Check | How to verify | Why it matters |
| --- | --- | --- |
| An **ACM certificate** for your domain, status *Issued*, in the **same account and us-east-1** | ACM console, or `aws acm list-certificates --region us-east-1` | The ALB can only use a certificate from its own account and Region. DNS can live in another account |
| **GPU vCPU quota**: at least 8 for G and VT (g6e, g7e), at least 16 for P (p5.4xlarge) | `aws service-quotas get-service-quota --service-code ec2 --quota-code L-DB2E81BA --region us-east-1` (G and VT); `--quota-code L-417A185B` (P) | g6e.2xlarge and g7e.2xlarge use 8 vCPUs; p5.4xlarge uses 16 |
| **Room for one more VPC, internet gateway, NAT gateway and Elastic IP** in the Region | Compare usage with quota, for example internet gateways: `aws ec2 describe-internet-gateways --region us-east-1 --query 'length(InternetGateways)'` against `aws service-quotas get-service-quota --service-code vpc --quota-code L-A4707A72 --region us-east-1` | The stack creates its own VPC, internet gateway, NAT gateway and Elastic IP. Our account was at its limit of 20 internet gateways and the first launch failed |
| **GPU capacity** you can actually get | Section 8, then `reserve-gpu.sh` | On-Demand GPU capacity is often sold out; see section 16 |
| **CloudShell** access | AWS console, top bar | Has the AWS CLI and your credentials, so nothing needs installing on a laptop |

**Optional, for the dashboard only:** the AWS CLI plus the Session Manager plugin on your laptop, to open the Bifrost dashboard through a tunnel. Installing the plugin needs admin rights on Windows. Everything else in this guide works without it.

**Where the quota codes come from.** `L-DB2E81BA` is *Running On-Demand G and VT instances* and `L-417A185B` is *Running On-Demand P instances*, both measured in vCPUs. Check them in the Service Quotas console under Amazon EC2.

## 11. Step-by-step deployment

Four steps in CloudShell take you from nothing to a running stack: reserve a GPU, edit one parameter file, create the stack, point DNS at it. Upload the repository (section 9) to CloudShell as a zip with **Actions, Upload file**, then run `unzip` and `cd` into it. Every command below runs from the repository's top folder.

### Step 1: Get GPU capacity

**For g6e or g7e**, reserve one instance with an On-Demand Capacity Reservation:

```bash
TYPE=g6e.2xlarge bash scripts/reserve-gpu.sh
# options go on the same line: TYPE=g7e.2xlarge AZS="us-east-1a us-east-1c" INTERVAL=60 bash scripts/reserve-gpu.sh
```

- **What it does:** checks which zones offer the type, then asks AWS for one instance in each zone every 60 seconds until one is granted. It reports success only once the reservation is **active**, because AWS can accept a request and mark it *failed* moments later.
- **Expected output:** `RESERVED cr-0… : g6e.2xlarge in us-east-1b (active - billing has started)`, followed by the exact `GpuAvailabilityZone` to use.
- **Cost:** a reservation bills at the On-Demand rate whether or not an instance runs in it, until you cancel it. While your instance runs inside it you pay once, not twice. Failed attempts cost nothing and disappear from the console after 60 minutes ([AWS docs](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/capacity-reservations-view.html)).

**For p5.4xlarge**, buy an **EC2 Capacity Block for ML** for the dates you need instead (EC2 console, Capacity Reservations, Purchase Capacity Blocks for ML). AWS guarantees the H100 for that window, which can start up to 8 weeks ahead ([AWS docs](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-capacity-blocks.html)).

### Step 2: Edit the parameter file

Open `params-g6e.json`, `params-g7e.json` or `params-p5.json` in the cloudformation/params folder and set these values. Leave everything else as it is.

| Parameter | Set to | Why |
| --- | --- | --- |
| `CertificateArn` | Your Issued ACM certificate ARN | HTTPS on the ALB. Alternative: leave it empty and set `HostedZoneId` if Route 53 is in this account, and the stack creates and validates a certificate |
| `DomainName` | The name callers will use, matching the certificate | Bifrost's allowed origin and the endpoint URL |
| `GpuAvailabilityZone` | The zone step 1 printed | The instance must start where the reservation is |
| `GpuHostPlacement` | `private-a` | Puts the instance in `GpuAvailabilityZone`'s private subnet |
| `SecondAvailabilityZone` | Any other zone | The ALB needs a second subnet; nothing else runs there |
| `ProjectName` | Unique per stack | Log groups and names; two stacks can't share one |
| `AlarmEmail` | Your email address | Alarm notifications. Empty means no emails. Confirm the SNS subscription email AWS sends you |

The defaults that matter, and why they stay on the first launch:

- **`MaxNumSeqs`** `0`: chooses 64 parallel requests on g6e and 128 on g7e or p5.
- **`MtpSpeculativeTokens`** `0`: multi-token prediction stays off until you've measured it (section 14).
- **`AllowModelStagingWrite`** `true`: lets the first boot copy the weights to S3.
- **`AmiIdOverride`** empty: uses the latest Deep Learning AMI on the first launch. Pin it before any update (section 14).

### Step 3: Create the stack

```bash
STACK=freyr-llm-qwen-stack        # one name per stack, e.g. freyr-llm-qwen-p5 for a second one
# The template is about 62 KB, over the 51,200-byte limit for --template-body, so stage it in S3 first.
BUCKET=cfn-templates-$(aws sts get-caller-identity --query Account --output text)-us-east-1
aws s3api head-bucket --bucket $BUCKET 2>/dev/null || aws s3 mb s3://$BUCKET --region us-east-1
aws s3 cp cloudformation/llm-qwen-gpu-bifrost.yaml s3://$BUCKET/llm-qwen-gpu-bifrost.yaml
aws cloudformation create-stack --region us-east-1 --stack-name $STACK \
  --template-url https://$BUCKET.s3.us-east-1.amazonaws.com/llm-qwen-gpu-bifrost.yaml \
  --parameters file://cloudformation/params/params-g6e.json --capabilities CAPABILITY_IAM
aws cloudformation wait stack-create-complete --region us-east-1 --stack-name $STACK && echo DONE
```

- **Why S3:** CloudFormation accepts at most 51,200 bytes inline and 1 MB from S3 ([CloudFormation quotas](https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/cloudformation-limits.html)). The console's *Upload a template file* option stores it in S3 for you, which is why the PoC launches from the console worked.
- **Why `CAPABILITY_IAM`:** the stack creates IAM roles without fixed names.
- **How long:** 25 to 45 minutes on a first launch. Most of it is the 29 GB model download and S3 copy, then about 2.3 minutes for vLLM to start (138 s on g6e, of which `torch.compile` is 38 s).
- **What decides success:** after the model is in place, the instance polls vLLM and Bifrost `/health` for up to 30 minutes, then signals CloudFormation SUCCESS or FAILURE. CloudFormation itself waits at most 75 minutes for that signal (`CreationPolicy` `PT75M`). On FAILURE or timeout the stack rolls back and deletes everything, so you're never left paying for a broken GPU.

### Step 4: Point DNS at the load balancer

```bash
aws cloudformation describe-stacks --region us-east-1 --stack-name $STACK \
  --query "Stacks[0].Outputs[?OutputKey=='DnsRecordToCreate'].OutputValue" --output text
```

Create that CNAME in the account that owns the Route 53 zone. A new stack means a new load balancer, so update the CNAME every time you recreate the stack.

## 12. Inside the EC2 instance: what the bootstrap installs and why

Nothing is installed by hand. The template's user data runs once, on first boot, as root, and logs everything to `/var/log/llm-bootstrap.log` and the `/<ProjectName>/bootstrap` CloudWatch log group. Any failing command stops it and signals failure to CloudFormation (`set -euo pipefail` plus an `ERR` trap).

```mermaid
flowchart LR
    A[DLAMI boots<br/>driver + CUDA ready] --> B[apt: Docker, Compose,<br/>NVIDIA runtime]
    B --> C[Format + mount<br/>EBS at /opt/models]
    C --> D{Weights in S3?}
    D -- no --> E[hf download to NVMe<br/>then copy to S3]
    D -- yes --> F[Sync S3 to /opt/models]
    E --> F
    F --> G[Secrets loader +<br/>Bifrost config.json]
    G --> H[docker-compose.yml +<br/>systemd llm-stack]
    H --> I[CloudWatch agent]
    I --> J{vLLM + Bifrost<br/>healthy in 30 min?}
    J -- yes --> K[Signal SUCCESS<br/>stack completes]
    J -- no --> L[Signal FAILURE<br/>stack rolls back]
```

**The starting image.** The AWS *Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)*, resolved from the public SSM parameter `/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id`. It already contains the NVIDIA driver (595.91.07 on our instance), CUDA and the NVIDIA Container Toolkit, so the GPU works on first boot without compiling drivers.

1. **Packages.** Installs Docker, Docker Compose v2 and Python venv with `apt-get -o DPkg::Lock::Timeout=900`, then runs `nvidia-ctk runtime configure --runtime=docker` so containers can use the GPU. *Why the lock timeout:* Ubuntu's automatic updates hold the apt lock right after boot; without the wait, the install fails at random.
2. **Data volume.** Finds the second EBS disk, formats it as XFS only if it's blank, and mounts it at `/opt/models` through `/etc/fstab` by UUID with `nofail`. *Why:* a re-attached volume keeps its data, and the mount survives reboots and stop/start.
3. **Model weights.** If the pinned commit isn't in S3 yet, it downloads it from Hugging Face with the `hf` CLI into the fast local NVMe disk, copies it to `s3://<bucket>/models/<repo>/<commit>/`, and deletes the staging copy. The first boot then syncs S3 to `/opt/models`. *Why S3 is the source of truth:* later boots reuse the copy on the data volume, and a new stack loads from S3 instead of the internet, and the S3 gateway endpoint keeps that traffic private and free of NAT charges.
4. **Secrets loader.** Writes `/opt/llm/load-secrets.sh`, which reads the Bifrost encryption key, setup token and admin password from Secrets Manager into `/run/llm/bifrost.env` (mode 600, in memory only). It runs before every service start, so no secret is ever written to the disk or the template.
5. **Bifrost configuration.** Writes `/opt/models/bifrost/config.json` once, owned by UID 1000 (Bifrost's user). It holds `env.` references, not secret values, and sets four things:
   - the vLLM provider at `http://vllm:8000` with `allow_private_network: true`;
   - `client.enforce_auth_on_inference: true`, so every model call needs a virtual key;
   - `governance.auth_config`, the admin account, with its password read from the environment;
   - a SQLite config store at `./config.db`, so keys and settings survive restarts.
6. **Containers.** Writes `/opt/llm/docker-compose.yml` with two services on a private Docker network named `llm`:
   - **vllm**: the pinned `vllm/vllm-openai` image; GPU access; port 8000 bound to `127.0.0.1` only; weights mounted read-only; compile cache on the data disk; logs to CloudWatch. Its flags: `--max-model-len`, `--gpu-memory-utilization 0.90`, `--kv-cache-dtype`, `--reasoning-parser qwen3`, `--enable-auto-tool-choice`, `--tool-call-parser qwen3_xml`, `--max-num-seqs`, and optionally `--speculative-config` for MTP.
   - **bifrost**: the pinned `maximhq/bifrost` image; port 8080, which only the ALB's security group can reach; its data directory on the EBS volume; secrets from the in-memory env file.
7. **systemd unit `llm-stack`.** Runs the secrets loader, then `docker compose up -d`. *Why:* the services come back by themselves after every reboot or stop/start.
8. **CloudWatch agent.** Ships the bootstrap log and publishes GPU utilisation, GPU memory, memory and disk metrics, which feed the GPU-saturation and model-disk alarms.
9. **Health gate.** Polls `localhost:8000/health` and `localhost:8080/health` every 10 seconds for up to 30 minutes, then calls `cfn signal-resource` with SUCCESS or FAILURE. *Why:* the stack only completes when the endpoint actually works.

**Measured start-up times on g6e.2xlarge:** weights loaded into the GPU in 11.6 s; `torch.compile` took 37.8 s; the whole vLLM start took 138 s; later restarts reuse the compile cache.

**Why containers instead of installing vLLM directly?** The pinned images bundle a matching PyTorch, CUDA libraries and kernels. Upgrading or rolling back is a change to one image tag, and the host only needs the driver and the container toolkit, which the AMI already has.

## 13. After deployment: verify, create keys, make the first call

Open a shell on the instance with **EC2 console, select the instance, Connect, Session Manager**. The browser shell starts as `ssm-user` in plain `sh` in a folder you can't write to, so first run `sudo -i` and `cd /root`. Then create the two scripts from the repository with the same names: `nano verify-llm-host.sh`, paste the file, save with **Ctrl+O, Enter**, exit with **Ctrl+X**. Do the same for `llm-admin.py`.

### 1. Verify every layer (SSM)

```bash
bash verify-llm-host.sh        # already root after sudo -i; bash, not sh
```

Expect **PASS** on every line. The script checks:

- the GPU, driver and Docker NVIDIA runtime;
- the disks and the model files;
- both containers, with restart counts;
- vLLM health and a test answer;
- Bifrost health, the key requirement and the admin lock;
- NAT egress, the instance role and the CloudWatch agent.

If vLLM shows FAIL with *"first load takes several minutes"*, it's still compiling. Wait two minutes and run it again.

### 2. Check the gateway and create one key per caller (SSM)

```bash
sudo python3 llm-admin.py status                          # all PASS, virtual keys: 0
sudo python3 llm-admin.py create my-laptop agent-app rag-app
sudo python3 llm-admin.py list                            # id, name, active
sudo python3 llm-admin.py revoke <id>                     # remove one caller
```

- **What `create` does:** for each name it creates a virtual key limited to the served model, sends one test question through Bifrost, and prints the `sk-bf-…` key **once**. Store each key in that application's secret store. Budgets and rate limits aren't set by the script; add them per key in the dashboard if you need them.
- **The admin password:** `llm-admin.py` reads it on the instance, so you never type it.
- **The dashboard (optional):** needs the AWS CLI and the Session Manager plugin on your laptop. Get the commands from the stack outputs, then open `http://localhost:8080`:

```bash
for k in DashboardTunnelCommand AdminPasswordCommand BifrostAdminUser; do
  aws cloudformation describe-stacks --region us-east-1 --stack-name $STACK \
    --query "Stacks[0].Outputs[?OutputKey=='$k'].OutputValue" --output text; done
```

### 3. Make the first call from anywhere

```bash
KEY='sk-bf-...'; URL='https://<your-domain>/v1/chat/completions'
curl -s "$URL" -H 'Content-Type: application/json' -H "x-bf-vk: $KEY" -d '{
  "model": "vllm/qwen3.8-27b",
  "messages": [{"role": "user", "content": "Say hello in one line"}],
  "max_tokens": 2048, "reasoning_effort": "low" }'
curl -s -o /dev/null -w 'no key: %{http_code}\n' "$URL" -H 'Content-Type: application/json' \
  -d '{"model":"vllm/qwen3.8-27b","messages":[{"role":"user","content":"hi"}]}'   # expect 401
```

- **Streaming:** add `"stream": true` to the body and `-N` to curl, and words arrive as they're generated.
- **The answer:** it's in `choices[0].message.content`, and the model's thinking is in `choices[0].message.reasoning`.

### 4. From application code

```python
from openai import OpenAI
client = OpenAI(base_url="https://<your-domain>/v1", api_key="sk-bf-...")   # sent as a Bearer token; Bifrost accepts it
r = client.chat.completions.create(
    model="vllm/qwen3.8-27b",
    messages=[{"role": "user", "content": "Summarise ..."}],
    max_tokens=16384,            # thinking counts toward this
    reasoning_effort="low",      # low | medium (xhigh: see section 2)
    stream=True,
)
for chunk in r:
    print(chunk.choices[0].delta.content or "", end="")
```

Any OpenAI-compatible framework (LangChain, Strands Agents, LlamaIndex, the OpenAI Agents SDK) works by changing only the base URL, key and model name.

### 5. The Qwen Chat app (optional, on a laptop)

The repository's `qwen-chat` folder is a local chat interface:

- streaming answers, with the thinking shown in a collapsible panel;
- the thinking level, plus max tokens, temperature, top-p and a system prompt;
- file attachments: text, code, CSV, JSON and PDF, and images when turned on;
- a timing summary under every answer: first token, thinking, answer, total, tokens and tokens/s.

Run `npm start` in that folder and open `http://localhost:5178`. It needs Node.js 18.17 or newer and has no dependencies. Set its context window in Settings to match the stack: 65,536 on g6e, 131,072 on g7e or p5. Request bodies are capped at 20 MB, the gateway's limit. It runs a small local server for two reasons: Bifrost only accepts browser requests from your own domain (CORS), and the server keeps your key out of the browser.

## 14. Operations

Day to day there are three jobs: start and stop the GPU to control cost, watch the alarms, and change the stack without replacing the instance by accident.

### Daily routine

| Task | How | Notes |
| --- | --- | --- |
| Find the instance ID | `aws cloudformation describe-stacks --region us-east-1 --stack-name $STACK --query "Stacks[0].Outputs[?OutputKey=='GpuHostInstanceId'].OutputValue" --output text` | Used by the next two rows |
| Stop for the night | `aws ec2 stop-instances --region us-east-1 --instance-ids <id>` | Stops GPU billing. Disks, model and keys are kept |
| Start in the morning | `aws ec2 start-instances --region us-east-1 --instance-ids <id>` | Services start by themselves; vLLM is ready in about 3 minutes (compile cache reused) |
| Automatic office hours | Set `EnableOfficeHours=true` (default schedule 09:00 to 21:00 Mon to Fri, Asia/Kolkata) | EventBridge Scheduler stops and starts the instance |
| Health check | `bash verify-llm-host.sh` and `python3 llm-admin.py status` (SSM, after `sudo -i`) | After every start |
| Logs | CloudWatch log groups `/<ProjectName>/vllm`, `/bifrost`, `/bootstrap` | Same as `docker logs`, kept for `LogRetentionDays` |

**What still bills while the instance is stopped:**

- A capacity reservation, at the full On-Demand rate, until you cancel it.
- The NAT gateway, the load balancer and the WAF web ACL, which charge by the hour or month.
- EBS storage, S3 and Secrets Manager.

For a PoC that idles for days, delete the stack and recreate it. The S3 bucket is emptied on delete, so the model downloads again on the next launch.

### Alarms (SNS email, set `AlarmEmail`)

| Alarm name | Fires when | First thing to check |
| --- | --- | --- |
| `<ProjectName>-target-unhealthy` | The ALB can't reach Bifrost `/health` | Is the instance running? `docker ps` on it |
| `<ProjectName>-target-5xx` | 10 or more 5xx responses from Bifrost in 5 minutes | Bifrost log; a burst usually means vLLM is restarting or rejecting requests |
| `<ProjectName>-alb-502` | 5 or more 502s generated by the ALB itself in 5 minutes | Usually a target keep-alive shorter than the ALB idle timeout; lower `AlbIdleTimeoutSeconds` |
| `<ProjectName>-host-status` | EC2 status check failed | EC2 console, instance status |
| `<ProjectName>-gpu-saturated` | GPU utilisation above 90% on average over three 5-minute periods | Too many parallel requests; consider p5 or a second instance |
| `<ProjectName>-model-disk` | `/opt/models` above 85% full | Old model revisions or a growing compile cache |

### Making answers faster: multi-token prediction

Test MTP on the running instance before turning it on in the template (SSM):

```bash
sudo sed -i "s|^\(\s*\)- --max-num-seqs=\(.*\)$|&\n\1- '--speculative-config={\"method\":\"mtp\",\"num_speculative_tokens\":2}'|" /opt/llm/docker-compose.yml
cd /opt/llm && sudo docker compose up -d vllm
until curl -sf localhost:8000/health >/dev/null; do sleep 10; done
curl -s localhost:8000/metrics | grep -E 'spec_decode_num_(accepted|draft)_tokens_total'
```

- **Is it working?** Ask the same question as before and compare the speed. Accepted tokens should be a large share of draft tokens; the vLLM recipe reports an acceptance rate of about 0.77 for this model in FP8 on other GPUs ([vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B)).
- **If vLLM won't start:** remove the added line and run `docker compose up -d vllm` again, or lower `--max-num-seqs` to 32.
- **To keep it:** set `MtpSpeculativeTokens=2` for the next stack.

### Changing the stack safely

Two things replace the GPU instance on a stack update, which means a new disk, a fresh model sync and an empty Bifrost (no keys):

1. **A newer AMI.** The AMI parameter is re-read from AWS's public SSM parameter on every update. Before any update, set `AmiIdOverride` to the `AmiInUse` output.
2. **Any parameter used in the launch template.** That includes `InstanceType`, `GpuHostPlacement`, the image tags, the model and its settings (`MaxModelLen`, `MaxNumSeqs`, `MtpSpeculativeTokens`, `GpuMemoryUtilization`, `KvCacheDtype`, `ToolCallParser`, `ServedModelName`), and also `DomainName`, `BifrostAdminUser` and `ProjectName`. All of them are baked into the instance definition or its startup script.

Always update through a change set and check that `GpuHost` doesn't show `Replacement: True` before you run it. To change a vLLM flag on a running instance without replacing it, edit `/opt/llm/docker-compose.yml` and run `docker compose up -d vllm`, as in the MTP test. Record that change in the template too, so the next stack gets it.

After the first successful launch, also set `AllowModelStagingWrite=false`, which removes the instance's write access to the model bucket.

## 15. Security model

A caller from the internet can reach exactly one thing: `/v1/*` through WAF and the ALB, and only with a valid virtual key. Everything else is closed by network rules, not by convention.

| Layer | Control | What it stops |
| --- | --- | --- |
| Network | GPU instance in a private subnet, no public IP; host security group allows port 8080 only from the ALB's security group; egress limited to 80 and 443 | Direct access to the instance, vLLM or the admin API from anywhere else |
| Edge | AWS WAF: AWS managed common rules, known bad inputs, Amazon IP reputation list, rate limit per IP (default 2000 per 5 minutes), optional allow-list of caller CIDRs. Four common-rule checks on request bodies are set to **count only** so long prompts aren't blocked: `SizeRestrictions_BODY`, `CrossSiteScripting_BODY`, `GenericLFI_BODY`, `GenericRFI_BODY` | Scanners, known exploit patterns, floods. Body-based XSS, LFI and RFI checks only log; that's acceptable because bodies go to a model, not a web app |
| Transport | TLS 1.2+ security policy on the ALB with your ACM certificate; HTTP redirects to HTTPS | Eavesdropping, key theft in transit |
| Routing | ALB forwards only `/v1/*`; everything else gets 404 | The Bifrost dashboard and `/api/*` are never exposed |
| Gateway | `enforce_auth_on_inference: true`; one virtual key per caller, limited to the served model | Anonymous use; one leaked key affects one caller and can be revoked alone |
| Gateway admin | Admin password from Secrets Manager, applied at first boot; dashboard only through a Session Manager tunnel | Changes to keys or routing by anyone without the password and SSM access |
| SSRF guard | Bifrost blocks private IPs by default; only the vLLM provider may reach the private Docker network. Link-local addresses (169.254.x.x, the metadata service) stay blocked even then, per the comment on `AllowPrivateNetwork` in Bifrost v2.2.1 `core/schemas/provider.go` | A crafted provider URL reaching internal services |
| Identity | IAM instance role, **scoped** rather than strict least privilege: list the model bucket and read `models/*`; write `models/*` only while `AllowModelStagingWrite=true`; read 4 named secrets; signal its own stack; plus the AWS managed policies `AmazonSSMManagedInstanceCore` (Session Manager) and `CloudWatchAgentServerPolicy` (logs and metrics, not limited to this stack's log groups) | Broad AWS access from a compromised container. Set `AllowModelStagingWrite=false` after the first launch |
| Instance metadata | IMDSv2 required, hop limit 1 | Containers stealing the role's credentials through IMDSv1 |
| Secrets | Secrets Manager only; loaded into an in-memory file (mode 600) before each start; never in the template, AMI or logs | Secrets on disk, in Git or in CloudFormation history |
| Admin access | Session Manager only, no SSH keys, no inbound ports | Lost keys and open port 22 |
| Data | `disable_content_logging: true` in Bifrost; S3 buckets block public access and deny non-TLS requests; EBS volumes encrypted | Prompts and answers stored in logs; data exposure at rest |

**What you still own:**

- **Key hygiene:** create one key per application, store it in that app's secret store, and revoke and recreate it if it's ever exposed. A key appeared in a screenshot during this PoC.
- **The caller allow-list:** set `AllowedCallerCidrs` if every caller comes from known networks.
- **Prompt-level safety:** the gateway controls who calls the model, not what they ask. Add guardrails in the application, or in Bifrost's plugin chain, if your use case needs them.

## 16. Troubleshooting and lessons learned

Every problem below happened during the PoC between 8 and 19 October 2026. Each one is now either fixed in the template and scripts or handled by a documented step. They're listed in the order you'd meet them.

| Stage | Symptom | Root cause | Evidence | Fix |
| --- | --- | --- | --- | --- |
| Stack create | *The maximum number of internet gateways has been reached* | Region quota for internet gateways was full (20 used, 2 of them detached) | Error from CloudFormation; IGW list in the VPC console | Delete detached gateways or raise quota `L-A4707A72` |
| Stack create | *We currently do not have sufficient g7e.2xlarge capacity* in 1a and 1c, later the same for g6e | No free GPU hosts in that zone; On-Demand launches aren't queued | CloudFormation event `InsufficientInstanceCapacity` | Reserve first with `reserve-gpu.sh` (section 11), or use a Capacity Block for p5.4xlarge |
| Capacity | Reservation script stopped with *Unsupported* in us-east-1b | g7e.2xlarge isn't offered in 1b or 1d at all | `describe-instance-type-offerings` returns 1a and 1c only | Script now checks offerings first and drops unsupported zones |
| Capacity | Many **Failed** reservations listed in the console | AWS records accepted requests that it then can't fill; they're visible for 60 minutes and never billed | [AWS docs on reservation states](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/capacity-reservations-view.html) | None needed; the script now waits for the real state and only reports *active* |
| Capacity | `AZS="…"` had no effect | Set on its own line, so the script never saw it | Shell behaviour | Put options on the same line: `TYPE=… AZS=… bash reserve-gpu.sh` |
| Bootstrap | Bifrost restarting: *failed to unmarshal sqlite config: unexpected end of JSON input* | `config_store` had no `config.path` | Bifrost log; [Bifrost storage docs](https://docs.getbifrost.ai/deployment-guides/config-json/storage) | `"config": {"path": "./config.db"}` in the template |
| Bootstrap | vLLM restarting: *max\_num\_seqs (256) exceeds available Mamba cache blocks (165)* | vLLM's default of 256 parallel requests needs more fixed per-request state than fits beside the weights on 48 GB | vLLM traceback in `docker logs vllm` | `MaxNumSeqs=0` sets 64 on g6e and 128 on 80–96 GB GPUs |
| Bootstrap | vLLM warns *Weight-only FP8 … Marlin kernel* on L40S | vLLM falls back to weight-only FP8 for this block-scaled checkpoint on the L40S (Ada), although Ada supports FP8 in general | vLLM log | Expected; correct output, slower prefill. Use p5 or g7e for native FP8 |
| Gateway | HTTP 502 *connection to private IP 172.18.0.2 is not allowed* | Bifrost blocks private IPs by default to prevent server-side request forgery; vLLM sits on a private Docker network | Bifrost response body; [Bifrost issue #4138](https://github.com/maximhq/bifrost/issues/4138); `allow_private_network` in v2.2.1 source | `allow_private_network: true` on the vLLM provider only |
| Gateway | Admin API reachable without a password | No admin account existed yet | Bifrost [security guide](https://docs.getbifrost.ai/deployment-guides/how-to/security-best-practices) | Admin account created from Secrets Manager on first boot |
| Gateway | Locking via API returned *log\_retention\_days must be at least 1* | Sending back current settings included a zero value Bifrost rejects on save | API response | Avoided; the template now locks it from `config.json` |
| Gateway | `llm-admin.py status` crashed: *object of type 'NoneType' has no len()* | Bifrost returns `virtual_keys: null` when there are none | Traceback | Fixed in the script |
| Deploy | `create-stack --template-body` fails for this template | Template is about 62 KB; the inline limit is 51,200 bytes | [CloudFormation quotas](https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/cloudformation-limits.html) | Upload to S3 and use `--template-url`, or use the console |
| Verify | `[[: not found` and false FAILs | Script run with `sh` instead of `bash` | Shell output | `sudo -i, then bash verify-llm-host.sh`; the script now refuses `sh` |
| Requests | HTTP 500 with `reasoning_effort: high` | The Qwen3.8 chat template accepts only `low`, `medium`, `xhigh` | [Qwen3.8 issue #217](https://github.com/QwenLM/Qwen3.8/issues/217) | Use `low` or `medium` |
| Requests | Empty answer, `finish_reason: length` | `max_tokens` used up by thinking | Response body | Raise `max_tokens` (16384 is a good default) or lower `reasoning_effort` |
| Requests | 401 on a test that should succeed | The placeholder `<sk-bf-...>` was sent literally | Shell history | Paste the real key |
| Client | Browser page can't call the endpoint | Bifrost allows browser requests (CORS) only from your domain | Bifrost `allowed_origins` | Call it server-side; the Qwen Chat app does this through its local server |
| Laptop | *SessionManagerPlugin is not found* | The plugin isn't installed, and installing it needs admin rights | AWS CLI error | Use the browser Session Manager; the dashboard is optional |
| Performance | Long answers take minutes (4,516 tokens in 181 s) | Decode is limited by memory bandwidth: about 25 tokens/s on L40S | Timing summary in the chat app | MTP (section 14), shorter answers, or a faster GPU (section 8) |

## 17. Teardown

Cancel the capacity reservation, delete the stack, then remove the DNS record. The stack deletes cleanly with no manual emptying.

```bash
STACK=freyr-llm-qwen-stack
aws ec2 describe-capacity-reservations --region us-east-1 --filters Name=state,Values=active \
  --query 'CapacityReservations[].[CapacityReservationId,InstanceType,AvailabilityZone]' --output table
aws ec2 cancel-capacity-reservation --region us-east-1 --capacity-reservation-id <cr-...>
aws cloudformation delete-stack --region us-east-1 --stack-name $STACK
aws cloudformation wait stack-delete-complete --region us-east-1 --stack-name $STACK && echo DELETED
```

**What happens on delete:**

- **Buckets and snapshots:** a clean-up Lambda empties both S3 buckets, including every object version, and deletes the stack's EBS snapshots. Without it, a non-empty bucket would block the delete.
- **Everything else:** secrets, log groups, the data volume and the network are deleted too. Nothing uses termination protection or a Retain policy.
- **What you remove yourself:**
  - the capacity reservation;
  - the CNAME in the Route 53 account;
  - the `cfn-templates-<account>-us-east-1` bucket, if you created it for `--template-url`.

**A Capacity Block (p5) can't be cancelled.** It ends at its booked time, and you've already paid for it up front.

## 18. Glossary

| Term | Meaning |
| --- | --- |
| ALB | Application Load Balancer: AWS's HTTP(S) load balancer; ends TLS and routes by path |
| BF16 / FP8 | Number formats: 2 bytes and 1 byte per value. FP8 weights halve memory with near-identical quality for this model |
| Capacity Block | EC2 Capacity Blocks for ML: GPUs booked for fixed future dates, paid up front, no cancellation |
| Capacity Reservation (ODCR) | On-Demand Capacity Reservation: holds an instance slot in one zone until cancelled, billed at the On-Demand rate |
| Chat template | The wrapper text a model expects around each message; adds a fixed number of tokens per request |
| Continuous batching | Adding new requests to the running batch as others finish, instead of waiting for the batch to empty |
| Context window | The most tokens one request can hold: prompt, history, files, thinking and answer |
| CORS | The browser rule that stops a web page calling another site unless that site allows it |
| Decode | Generating the answer token by token; limited by GPU memory bandwidth |
| DLAMI | AWS Deep Learning AMI: Ubuntu with NVIDIA driver, CUDA and container toolkit preinstalled |
| Gated DeltaNet | The linear-attention layer type in Qwen3.8; keeps a fixed-size state per request |
| IMDSv2 | Instance Metadata Service v2: token-based access to instance credentials |
| KV cache | Stored keys and values for every past token in every attention layer; grows with context length |
| Marlin | GPU kernels that compute with low-bit weights in 16-bit maths; vLLM's fallback for this FP8 checkpoint on the L40S |
| `max_tokens` | The most tokens the model may write for one request, thinking included |
| MTP | Multi-token prediction: the model drafts several tokens and checks them in one pass |
| NAT gateway | Lets private instances make outbound connections without being reachable from outside |
| PagedAttention | vLLM's block-based KV-cache memory management |
| Prefill | Reading the prompt; limited by GPU compute; sets the time to first token |
| `reasoning_effort` | How long the model thinks: `low`, `medium`, `xhigh` for Qwen3.8 |
| Session Manager | AWS Systems Manager feature for shell access and port forwarding without SSH or open ports |
| SSRF | Server-side request forgery: tricking a server into calling internal addresses; why Bifrost blocks private IPs by default |
| Tensor parallelism | Splitting one model across several GPUs |
| Token | The unit models read and write; a word is often one to two tokens |
| Virtual key | A Bifrost-issued `sk-bf-…` key per caller, with its own model access, limits and usage record |
| vLLM | Open-source LLM inference server used here |
| WAF | AWS Web Application Firewall, attached to the ALB |

## 19. References

Every external fact in this guide comes from one of these pages, opened between 5 and 19 October 2026. Anything marked **Measured** in the text comes from our own instance.

**Model**

- [Qwen/Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B)
- [Qwen/Qwen3.8-27B-FP8 model card](https://huggingface.co/Qwen/Qwen3.8-27B-FP8)
- [config.json at the pinned commit](https://huggingface.co/Qwen/Qwen3.8-27B-FP8/raw/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a/config.json)
- [Qwen3.8 issue #217: reasoning\_effort "high" returns 500](https://github.com/QwenLM/Qwen3.8/issues/217)

**vLLM**

- [vLLM documentation](https://docs.vllm.ai/en/latest/)
- [vLLM recipe for Qwen3.8-27B](https://recipes.vllm.ai/Qwen/Qwen3.8-27B)
- [vLLM FP8 quantization](https://docs.vllm.ai/en/latest/features/quantization/llm_compressor/fp8/)
- [vLLM speculative decoding: MTP](https://docs.vllm.ai/en/latest/features/speculative_decoding/mtp/)
- [PagedAttention paper, arXiv 2309.06180](https://arxiv.org/abs/2309.06180)
- [vLLM launch blog](https://vllm.ai/blog/2023-06-20-vllm)
- [PyTorch Foundation welcomes vLLM](https://pytorch.org/blog/pytorch-foundation-welcomes-vllm/)
- [vLLM on GitHub](https://github.com/vllm-project/vllm)

**Inference performance**

- [NVIDIA: Mastering LLM Techniques: Inference Optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)
- [Databricks: LLM Inference Performance Engineering](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices)
- [Orca, OSDI '22](https://www.usenix.org/conference/osdi22/presentation/yu)
- [Anyscale: continuous batching](https://www.anyscale.com/blog/continuous-batching-llm-inference)

**Gateways**

- [Bifrost on GitHub](https://github.com/maximhq/bifrost)
- [Bifrost virtual keys](https://docs.getbifrost.ai/features/governance/virtual-keys)
- [Bifrost vLLM provider](https://docs.getbifrost.ai/providers/supported-providers/vllm)
- [Bifrost storage configuration](https://docs.getbifrost.ai/deployment-guides/config-json/storage)
- [Bifrost security best practices](https://docs.getbifrost.ai/deployment-guides/how-to/security-best-practices)
- [Bifrost issue #4138: private IP blocked](https://github.com/maximhq/bifrost/issues/4138)
- [Bifrost issue #6193: xhigh rewritten to high](https://github.com/maximhq/bifrost/issues/6193)
- [Bifrost benchmarks (vendor-published)](https://www.getmaxim.ai/bifrost/resources/benchmarks)
- [LiteLLM on GitHub](https://github.com/BerriAI/litellm)
- [LiteLLM virtual keys](https://docs.litellm.ai/docs/proxy/virtual_keys)
- [Gartner: Market Guide for AI Gateways (abstract)](https://www.gartner.com/en/documents/7051698)
- [AWS Guidance: Multi-Provider Generative AI Gateway](https://docs.aws.amazon.com/solutions/multi-provider-generative-ai-gateway-on-aws/)
- [AWS reference architecture blog](https://aws.amazon.com/blogs/machine-learning/streamline-ai-operations-with-the-multi-provider-generative-ai-gateway-reference-architecture)
- [Cloudflare AI Gateway](https://developers.cloudflare.com/ai-gateway/)
- [Kong AI Gateway](https://developer.konghq.com/ai-gateway/)

**AWS**

- [Amazon EC2 G6e](https://aws.amazon.com/ec2/instance-types/g6e/)
- [Amazon EC2 G7e](https://aws.amazon.com/ec2/instance-types/g7e/)
- [Amazon EC2 P5](https://aws.amazon.com/ec2/instance-types/p5/)
- [Capacity Reservations](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-capacity-reservations.html)
- [Capacity Reservation billing](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/capacity-reservations-pricing-billing.html)
- [Capacity Reservation states](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/capacity-reservations-view.html)
- [Capacity Blocks for ML](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-capacity-blocks.html)
- [Capacity Blocks billing](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/capacity-blocks-pricing-billing.html)
- [CloudFormation quotas](https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/cloudformation-limits.html)
- [SageMaker LMI v15 with vLLM](https://aws.amazon.com/blogs/machine-learning/supercharge-your-llm-performance-with-amazon-sagemaker-large-model-inference-container-v15/)
- [AWS Deep Learning Containers for vLLM](https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/dlc-vllm-ec2.html)
- [Amazon Rufus on Trainium and vLLM](https://aws.amazon.com/blogs/machine-learning/how-amazon-scaled-rufus-by-building-multi-node-inference-using-aws-trainium-chips-and-vllm)
- [p5.4xlarge pricing, us-east-1 (DoiT)](https://www.doit.com/compute/compute/aws/us-east-1/p5.4xlarge)
- [g6e.12xlarge pricing (DevZero)](https://www.devzero.io/instances/aws/g6e.12xlarge)

**Hardware**

- [NVIDIA L40S](https://www.nvidia.com/en-us/data-center/l40s/)
- [NVIDIA H100](https://www.nvidia.com/en-us/data-center/h100/)
- [NVIDIA RTX PRO 6000 Blackwell Server Edition](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/)

**Industry context**

- [a16z: generative AI in the enterprise, 2024](https://a16z.com/generative-ai-enterprise-2024/)
- [Menlo Ventures: 2025 mid-year LLM market update](https://menlovc.com/perspective/2025-mid-year-llm-market-update/)
- [Google Cloud: serve a model with vLLM on GKE](https://docs.cloud.google.com/kubernetes-engine/docs/tutorials/serve-gemma-gpu-vllm)
- [Red Hat AI Inference Server announcement](https://www.businesswire.com/news/home/20250520450429/en)
- [NVIDIA Dynamo feature matrix](https://docs.nvidia.com/dynamo/v1.2.0/resources/feature-matrix)
