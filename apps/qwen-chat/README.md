# Qwen Chat: local chat UI for your self-hosted model

A small chat app that runs on your laptop and talks to the Qwen3.8 endpoint behind Bifrost. It lets you:
- chat with streaming answers;
- see the model's thinking in a collapsible panel;
- choose the thinking level;
- attach files;
- get a timing summary under every answer: time to first token, thinking time, answer time, total, tokens and tokens/s.

No packages to install. It needs Node.js 18.17 or newer.

## Start it

```bash
cd qwen-chat
npm start
```

Open **http://localhost:5178**. The first time, Settings opens on its own:

| Field | Value |
|---|---|
| Endpoint URL | `https://llm.example.com/v1` (the full `/v1/chat/completions` URL also works) |
| Virtual key | Your `sk-bf-...` key. Create one on the instance with `sudo python3 llm-admin.py create my-laptop` |
| Send key as | `x-bf-vk header` (Bifrost's own header). `Authorization: Bearer` also works |
| Default model | `vllm/qwen3.8-27b` |

Click **Save and test connection**. The dot next to the model name turns green.

You can also copy `config.example.json` to `config.json` and edit it before starting.

Stop the app with **Ctrl+C**. Use another port with `PORT=6000 npm start`.

## Why there is a small local server

- **The key stays out of the browser.** The page talks to `localhost`, and the local server adds your key and forwards the request to the endpoint. The key is saved in `config.json`, readable only by your user, and the browser only ever sees a masked hint.
- **No CORS changes on the gateway.** Bifrost only accepts browser requests from your own domain. A server-side forward needs no change there.
- **Only your laptop can use it.** The server listens on `127.0.0.1` and refuses API calls from other web origins, so a website you visit can't use your key through it.

## What you can send

| Option | Where | What it sends |
|---|---|---|
| Thinking: Low / Medium / Extra high | Composer | `reasoning_effort: low / medium / xhigh`. **Model default** leaves it out. The model has no `high` level (it returns HTTP 500), so it isn't offered |
| Stream | Composer | `stream: true`, with `stream_options.include_usage` so token counts arrive at the end |
| Max tokens, temperature, top-p | Settings | `max_tokens`, `temperature`, `top_p`. Thinking counts toward max tokens; if an answer comes back empty, raise it |
| System prompt | Settings | A `system` message at the start of every chat |
| Files | Paperclip, drag and drop, or paste | See below |

**Files:**
- **Text, code, CSV, JSON, YAML, logs and similar:** read on your laptop and sent as `<document name="…">…</document>` before your message.
- **PDF:** text is extracted on your laptop with pdf.js, page by page. pdf.js is loaded from cdnjs the first time, so it needs internet. Scanned PDFs have no text and are refused with a message.
- **Images:** off by default. Turn on **Send attached images** only if the model you serve accepts images. They're sent as OpenAI `image_url` parts.
- **Word and Excel files** aren't read directly. Save them as PDF or CSV.
- **Limits:** 2 MB per text file and 8 MB per image. The composer shows a rough token count against the 65,536-token context.

## The timing summary under each answer

| Item | Meaning |
|---|---|
| First token | Send to first streamed byte. Includes network, WAF, ALB, Bifrost and the model's prompt processing |
| Thinking | First thinking token to first answer token |
| Answer | First answer token to the end |
| Total | Send to the last byte |
| Output / Input | Token counts reported by vLLM (`usage`) |
| Speed | Output tokens ÷ generation time |
| Gateway | Bifrost's own measured latency (`extra_fields.latency`) |

The bar above the numbers shows waiting, thinking and answering to scale.

**Non-streaming answers** arrive all at once, so they show only Response, Gateway, tokens and speed.

## Where things are stored

| What | Where |
|---|---|
| Chats and answer settings | Your browser's local storage for `localhost:5178`. **Delete all chats** is in Settings |
| Endpoint, key and model | `config.json` next to `server.js`. It's git-ignored |

Nothing is sent anywhere except your endpoint, plus cdnjs (pdf.js) and Google Fonts.

## If something goes wrong

| Message | Fix |
|---|---|
| The key was not accepted (401) | Wrong or revoked key. Create a new one with `llm-admin.py create` |
| The gateway could not reach the model (502/503) | The EC2 instance is stopped or still starting; vLLM needs about 3 minutes after a start |
| TLS certificate not trusted | A corporate proxy is intercepting HTTPS. Start with `NODE_EXTRA_CA_CERTS=/path/to/company-ca.pem npm start` |
| No answer yet, max tokens ran out | Raise Max tokens or lower the Thinking level, then **Retry** |
| Could not load the PDF reader | No internet access to cdnjs. Attach the text instead |
