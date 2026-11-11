# Pudgy inference API

Generate Pudgy Penguins videos with the trained model, from the dashboard's **Generate videos** page
or from your own code. The server runs on a GPU box that is switched on only when needed; when it is
off, the dashboard says **"No inference server available"** and still shows every saved video.

```
dashboard (Azure App Service)                         GPU box (this folder)
  Generate videos page ──── HTTPS + X-API-Key ──────►  server.py  (FastAPI, one job at a time)
        │                                                 ├─ musubi --interactive: the model stays loaded
        │ reads inference/server.json ◄── heartbeat 30 s ─┤  (eval_v10/common_v10: v8 merged + golden + E5)
        │ plays inference/jobs/<id>/video.mp4 ◄── upload ─┘
        └──────────── Azure container AZURE_DASHBOARD_CONTAINER (pudgy-dashboard) ────────────
```

**Model:** `v8merged` = the production setup: Wan2.2-I2V-A14B, v8 merged LoRA on the low-noise
expert, v2 golden on the high-noise expert, the E5 phone embedding (applied whenever the prompt holds
the registry phone phrase). 25 steps, guidance 5.0, shift 5, fp8. These are `common_v10.musubi_cmd`'s
settings, so a video here matches the evaluation. When v10 passes its gates, add it with
`MODELS=v8merged,v10@<step>`.

## Run it on the GPU box

```bash
# once: two lines in /workspace/Pudgy/.env (NAME=value; the file is read, never executed)
INFERENCE_API_KEY=<python3 -c 'import secrets; print(secrets.token_urlsafe(32))'>
AZURE_STORAGE_CONNECTION_STRING=...          # usually there already

bash /workspace/Pudgy/finetune/wan/infer_api/serve.sh start    # prints the public address
bash /workspace/Pudgy/finetune/wan/infer_api/serve.sh status
bash /workspace/Pudgy/finetune/wan/infer_api/serve.sh stop     # before switching the box off (optional)
```

What `start` does:
- On first use, it makes its own venv (`/workspace/infer_api_venv`; the trainer venv is untouched) and pulls the 29 preset start frames.
- It runs the server in tmux `infer` and opens a Cloudflare quick tunnel (HTTPS).
- It writes the server's address and a heartbeat to Azure, and the dashboard picks it up within about 20 s.
- A crash restarts the server, up to 3 times.

It refuses to start while a training run holds the GPU. The box must have the v10 setup (`v10_box.sh setup` and `pull`).

The tunnel address changes on every start, and that's fine because the dashboard reads it from Azure. For a fixed address, run a named Cloudflare tunnel (`CF_TUNNEL_TOKEN` at box launch) and start the server with `--public-url https://your.host` instead of `--tunnel`.

## Dashboard settings

The App Service needs `INFERENCE_API_KEY`, with the same value as the box (Configuration →
Application settings). It already has `AZURE_STORAGE_CONNECTION_STRING`. Optionally, set
`INFERENCE_API_URL` to pin a server address instead of reading the registration. Locally, both go in
the repo `.env`.

## Using the API directly

Interactive docs, where you can try every call with your key: `<address>/docs`.
Authentication: `X-API-Key: <key>` or `Authorization: Bearer <key>` on every call except `/v1/health`.

| Call | Does |
|---|---|
| `GET /v1/health` | up? `state`: `idle` / `busy` / `unavailable` (+ `reason`); queue length, running job, GPU |
| `GET /v1/info` | models, sizes, lengths, modes, limits, time estimates, prompt presets, preset start frames, self-checks |
| `POST /v1/prompt` | `{"guided": {...}}` or `{"prompt": "..."}` → the exact prompt the model gets, plus advice |
| `POST /v1/jobs` | queue a video (multipart: `params` JSON + `start_image` and/or `end_image` files) → the job |
| `GET /v1/jobs` · `GET /v1/jobs/{id}` | state, `stage`, `progress` 0–1, `queue_position`, `eta_seconds`, `warnings`, `error`, `result` |
| `GET /v1/jobs/{id}/video` · `/thumbnail` | the MP4 (H.264, 24 fps) · its first frame |
| `DELETE /v1/jobs/{id}` | cancel a queued or running job |
| `GET /v1/keyframes/{name}` | a preset start frame (PNG) |

`params` fields:

| Field | Values | Default |
|---|---|---|
| `guided` **or** `prompt` | guided: `characters` (`pax`/`polly`/`both`), `action` (see `/v1/info` presets: 14 single-character + 3 two-character + 2 phone), `setting` (`sofa`/`window`/`door`/`lounge`) or `setting_text`, optional `shot`, `details` · prompt: your own text, ≤ 1500 characters | — |
| `start_preset` or the `start_image` file | a name from `/v1/info` `keyframes`, or PNG/JPEG/WebP up to 20 MB | — |
| `size` | `portrait_720` (720×1280, reels), `portrait_480`, `landscape_720`, `landscape_480` | `portrait_720` |
| `frames` | frames per take: 33, 49, 81 (81 = 3.4 s at 24 fps) | 81 |
| `takes` | 1–3; takes 2 and 3 continue from the previous take's last frame (3 × 81 = 10 s) | 1 |
| `mode` | `chain` (continuous), `loop` (every take starts and ends on the start frame), `first_last` (1 take, start → your `end_image`) | `chain` |
| `seed` | 0–2147483644, or null for a random one (take k uses seed + k − 1) | random |
| `steps` | 10–50 | 25 |
| `negative_extra` | extra things to avoid, added to the built-in negative prompt | — |
| `model` | an available id from `/v1/info` | first available |

```python
import json, time, requests
API, H = "https://<address>", {"X-API-Key": "<key>"}
params = {"guided": {"characters": "polly", "action": "happy", "setting": "window"},
          "start_preset": "happy_polly", "size": "portrait_720", "takes": 2}
job = requests.post(f"{API}/v1/jobs", headers=H, data={"params": json.dumps(params)}).json()
while job["state"] in ("queued", "running"):
    time.sleep(30)
    job = requests.get(f"{API}/v1/jobs/{job['id']}", headers=H).json()
    print(job["state"], job["stage"], f"{job['progress']:.0%}", job["eta_seconds"], "s left")
if job["state"] == "done":
    open("polly_happy.mp4", "wb").write(requests.get(f"{API}/v1/jobs/{job['id']}/video", headers=H).content)
else:
    print(job["error"])
```

## The checks

- **Before a job is queued:**
  - Every field is validated, and a refusal says what's wrong (HTTP 422).
  - Text containing `--` or a line break is refused. musubi reads one line per take and splits options on ` --`, so such text could inject options or start extra generations.
  - The start or end frame must be a readable PNG/JPEG/WebP of 256–4096 px, no larger than 20 MB, and the same shape as the output (within 15%). A small difference is centre-cropped, with a warning.
  - Free-form prompts get advice: a missing style prefix, Pax or Polly named without their trained wording, a phone mentioned without the registry phrase (so E5 won't apply), or close-ups.
- **Capacity:**
  - The queue holds up to 10 jobs (HTTP 429 beyond that).
  - If the GPU holds more than 6 GB from another job (a training run, say) before our model loads, the server reports `unavailable` with the reason and refuses jobs (HTTP 503).
- **While a job runs:**
  - Progress comes from the sampler's step counter.
  - A take is stopped after 3× its estimate plus 20 min.
  - If the generator dies (out of memory, a bad input), the job fails with its last log lines, and the next job starts a fresh generator.
- **After a job:**
  - Each take's frame count and the joined video's are checked, with a warning on a mismatch.
  - The upload to Azure is verified by size, with a warning if it fails. The video can still be downloaded from the server.
- **Restarts and shutdown:**
  - Jobs survive a server restart: waiting jobs are re-queued, and a running job is marked `interrupted`.
  - A clean stop marks the registration `stopped`.
  - A box that is switched off stops its heartbeat, and the dashboard reports it offline after 2 minutes.
- **Estimates:** they start from the A100 long-form timing (about 30 min per 720×1280 take of 81 frames, plus about 5 min for the first model load) and are re-fitted from this server's own takes.

## Without a GPU

The mock backend makes placeholder videos (the start frame with a slow push-in), so the dashboard and the API can be tried on a laptop:

```bash
python server.py --backend mock --no-auth --port 8765 --work /tmp/infer --keyframes <folder of start frames>
INFERENCE_API_URL=http://127.0.0.1:8765 INFERENCE_API_KEY=x streamlit run dashboard/app.py
python -m pytest finetune/wan/infer_api/tests -q      # needs requirements.txt + httpx pytest
```
