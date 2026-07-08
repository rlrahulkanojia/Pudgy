# AI Video Generation — Model Pricing & Platform Guide

**Prepared for:** Pudgy Penguins AI Animation Sprint
**Date:** June 2026

---

## 1. Per-Model Pricing (Cost Per Second of Video)

These are the actual AI models. Multiple platforms offer access to the same models — pick your platform based on features, not model access.

| Model | Developer | Max Resolution | Max Duration | Cost/Second | Audio | Notes |
|-------|-----------|---------------|-------------|-------------|-------|-------|
| **Veo 3.1 Lite** | Google | 720p | 8s | $0.03–0.05 | No | Cheapest option for drafts |
| **Veo 3.1 Fast** | Google | 1080p | 8s | $0.10–0.15 | Yes | Best price/quality ratio |
| **Veo 3.1 Standard** | Google | 4K | 8s | $0.40 | Yes | Highest quality, cinematic grade |
| **Seedance 2.0 Fast** | ByteDance | 2K | 15s | $0.022 | Yes | Cheapest per-second of any model |
| **Seedance 2.0 Pro** | ByteDance | 2K | 15s | $0.247 | Yes | High quality, native audio |
| **Kling 3.0 Standard** | Kuaishou | 1080p | 15s | $0.084–0.126 | Optional | Audio adds ~50% cost |
| **Sora 2** | OpenAI | 720p | 12s | $0.10 | No | **Sunsetting Sept 24, 2026** |
| **Sora 2 Pro** | OpenAI | 1080p | 25s | $0.30–0.70 | No | Longest single generation; sunsetting |
| **Runway Gen-4.5** | Runway | 1080p | 10s | ~$0.50* | No | *Estimated from credit cost |
| **Runway Gen-4 Turbo** | Runway | 1080p | 10s | ~$0.10* | No | *Lower quality, much cheaper |
| **Pika 2.5** | Pika Labs | 1080p | 10s | ~$0.12–0.24* | No | *Estimated from credit cost |

*Runway/Pika don't publish per-second rates — estimates derived from credits-per-second / plan cost.*

**Disable audio when not needed** — it drops model costs by ~33-40% on Veo and Seedance.

---

## 2. Multi-Model Platforms (Access Multiple Models in One Place)

### Higgsfield AI — Best Multi-Model Aggregator
**Models included:** Kling 3.0, Veo 3.1, Sora 2, Seedance 2.0, + 11 more

| Plan | Monthly | Credits | Approx. Output |
|------|---------|---------|----------------|
| Starter | $15 | 70 | ~8 Kling clips OR ~1 Veo clip |
| **Plus** | $49 | 1,000 | ~142 Kling OR ~17 Veo clips |
| Ultra | $129 | 3,000 | ~428 Kling OR ~51 Veo clips |
| Business | $89/seat | 1,500/seat | Shared pool, min 2 seats |

**Cinematic features:**
- **Soul ID** — character consistency across multiple shots (same face/identity scene-to-scene)
- **Cinema Studio** — virtual production with lens types, focal lengths, optical physics
- **70+ camera presets** — Bullet Time, Crash Zoom, 360 Rotation, Dolly, etc.
- **VFX & Style Transfer** — Ghibli-style anime, explosions, transitions

**Caveat:** "Unlimited" marketing is misleading — only their oldest model (Soul V2) is truly unlimited. Premium models (Veo, Kling) consume credits fast.

### Runway — Best Creative Control for Filmmaking
**Models included:** Gen-4.5, Gen-4, Gen-4 Turbo, Veo 3.1, Kling 3.0, Seedance, FLUX

| Plan | Monthly | Credits | Gen-4.5 Video |
|------|---------|---------|---------------|
| Free | $0 | 125 (one-time) | 5 seconds |
| Standard | $12 | 625 | 25 seconds |
| **Pro** | $28 | 2,250 | 90 seconds |
| Unlimited | $76 | 2,250 + Explore Mode | 90s priority + unlimited slow queue |

**Cinematic features:**
- **Extend** — generate a clip, then extend it forward for shot continuity
- **Camera choreography** — timed beats, camera direction control
- **Multi-model switching** — same project, different models per shot
- **Character reference images** — feed reference to maintain identity across shots
- **Explore Mode** (Unlimited plan) — unlimited relaxed-rate generations for iteration

---

## 3. Single-Model Platforms

### Google Flow — Best Free Option + Clip Chaining
**Model:** Veo 3.1 + Imagen 3

| Plan | Monthly | Credits |
|------|---------|---------|
| **Free** | $0 | 50 daily (resets 24h) |
| AI Pro | $19.99 | 1,000 |
| AI Ultra | $249.99 | 25,000 |

**Cinematic features:**
- **Ingredients System** — node-based: characters, style, background, lighting as separate reusable assets
- **Clip chaining** — extend clips up to **~148 seconds** total via sequential generation
- **Gemini timeline awareness** — AI understands your full project timeline, not just current clip
- **Native audio** — dialogue with lip-sync, SFX, and music generated simultaneously
- **Gemini Omni Flash** — AI agent that can plan/reason through complex creative tasks

### Pika — Best for Stylized Effects & Transitions
**Model:** Pika 2.5, 2.2

| Plan | Monthly | Credits |
|------|---------|---------|
| Free | $0 | 80 |
| Standard | $8 | 700 |
| **Pro** | $28 | 2,300 |
| Fancy | $76 | 6,000 |

**Cinematic features:**
- **Pikascenes** — scene-to-scene morphing transitions
- **Pikadditions/Pikaswaps** — add/swap elements in existing video
- **Pikaffects** — stylized visual effects
- Best for transitions and effects between shots, not sustained character animation

### Kling — Best Temporal Consistency
**Model:** Kling 3.0

| Plan | Monthly | Credits |
|------|---------|---------|
| Free | $0 | 66 daily |
| Standard | $7–10 | 660 |
| **Pro** | $26–37 | 3,000 |
| Premier | $65–92 | 8,000 |
| Ultra | $128–180 | 26,000 |

**Cinematic features:**
- **Image-to-Video** — feed last frame of Clip A as first frame of Clip B for shot continuity
- Strong temporal consistency for multi-shot sequences
- Native audio generation (adds ~50% credit cost)

**Caveat:** Failed generations consume credits with no refund. Credits don't roll over.

### Seedance 2.0 (via Dreamina) — Cheapest Per-Second
**Model:** Seedance 2.0

| Plan | Monthly | Credits |
|------|---------|---------|
| Free | $0 | 225 daily (shared across tools) |
| Basic | $18 | ~1,900 |
| Standard | $48 | ~5,300 |
| Advanced | $84 | ~10,400 |

**Cinematic features:**
- **15-second native generation** — longest single clip among budget options
- **Native audio** — generated with video
- Cheapest API rate at $0.022/sec (Fast tier)

### Midjourney — Best Painterly Aesthetic
**Model:** Midjourney V8.1 (image) + Video V1

| Plan | Monthly | Fast GPU Hours |
|------|---------|---------------|
| Basic | $10 | 3.3 hrs |
| Standard | $30 | 15 hrs |
| **Pro** | $60 | 30 hrs + unlimited Relax for video |
| Mega | $120 | 60 hrs |

**Cinematic features:**
- Painterly, atmospheric visual style (unique among all platforms)
- Video pulls from same GPU hours as images — ~26 GPU min per HD video batch
- **Pro or Mega required** for practical video generation (unlimited Relax mode)
- No free tier

### Sora 2 (API only) — Sunsetting Sept 2026
**Model:** Sora 2, Sora 2 Pro

| Tier | Resolution | Cost/Second |
|------|-----------|-------------|
| Sora 2 | 720p | $0.10 ($0.05 batch) |
| Sora 2 Pro | 720p | $0.30 ($0.15 batch) |
| Sora 2 Pro | 1080p | $0.70 ($0.35 batch) |

**Consumer app discontinued April 26, 2026. API sunsetting September 24, 2026. Not recommended for new projects.**

---

## 4. Platforms Already Tested by Pudgy Team

| Platform | Result | Issue |
|----------|--------|-------|
| **Kling 3.0** | Tested | Lacked continuity and consistency for Pudgy IP |
| **Sora** | Tested ("best ones") | Lacked continuity and consistency; now sunsetting |
| **Seedance** | Tested ("best ones") | Lacked continuity and consistency |
| **Krea** | Currently evaluating | Real-time generation, good for rapid iteration |
| **Higgsfield** | Tested ("whatever was on Higgsfield") | Not specified |

**Common failure across all tested platforms:** None maintained character identity (penguin proportions, accessories, scarf colors) consistently across shots. This is the fundamental limitation of general-purpose models for IP-specific animation.

---

## 5. Quick Decision Matrix

| Need | Best Option | Monthly Cost |
|------|------------|-------------|
| **Free experimentation** | Google Flow (free tier) | $0 |
| **Test all models in one place** | Higgsfield Plus | $49 |
| **Maximum creative control** | Runway Pro | $28 |
| **Cheapest per-second generation** | Seedance 2.0 via API ($0.022/s) | Pay-per-use |
| **Clip chaining for longer videos** | Google Flow (148s chaining) | $0–$19.99 |
| **Character consistency across shots** | Higgsfield Soul ID | $49+ |
| **Painterly/artistic style** | Midjourney Pro | $60 |
| **Budget production stack** | Flow (free) + Kling Pro (~$30) | ~$30 |
| **Full experimentation stack** | Flow + Higgsfield + Runway | ~$77 |

---

*Sources: [Higgsfield](https://higgsfield.ai/pricing), [Runway](https://runwayml.com/pricing), [Google Flow](https://labs.google/fx/tools/flow), [Pika](https://pika.art/pricing), [Kling](https://www.eesel.ai/blog/kling-ai-pricing), [Seedance](https://seedancegen.com/pricing), [Midjourney](https://docs.midjourney.com/hc/en-us/articles/27870484040333), [Sora](https://costgoat.com/pricing/sora), [Veo 3.1](https://www.aifreeapi.com/en/posts/veo-3-1-pricing)*
