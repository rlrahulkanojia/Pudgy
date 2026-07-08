# Pudgy Penguins: How We'll Build Your Custom Animation Engine

A plain-language overview of the approach, what it will and won't do, and how your team stays in the loop. The goal is an engine that produces brand-faithful Pax and Polly animation, starting with looping GIFs and scaling to longer shorts.

## The idea in one line
We train a model on *your own* art and animation so it learns Pax, Polly, and your props/worlds. An artist then sets up the opening frame of a shot, and the engine animates from there, finishing in your style.

## What makes this different from the AI tools you tried
The off-the-shelf tools (Seedance, Kling, etc.) only ever saw generic internet video, so they guess at your style and drift off-model. This engine is **trained on Pudgy's actual library**, so the characters stay on-model and the motion follows your animation principles. It is also **animator-led, not push-button**: your team composes the key frame and reviews every output, so you keep creative control instead of fighting a black box.

---

## How it works (4 steps)

**1. Prepare the data.** We take the clips and assets you send (see the Data Requirements doc) and turn them into a clean, consistent training set: short single-action clips of Pax and Polly, a mix of plain-background and in-scene shots, plus reference images of each character and prop.

**2. Train the model.** The engine learns your characters, props, and style from that set. We run several training passes and pick the best one by eye, checking that characters stay on-model and motion looks right.

**3. Build the production pipeline.** We assemble the generation flow so an artist can drop in a composed opening frame and get back a finished, **seamless looping GIF at your 24 fps standard**. (The 24 fps and smooth-loop handling are built in.)

**4. Produce and review.** We generate batches of GIFs, your animation director scores them against an agreed quality bar, and we refine until they pass.

---

## How your team stays in control
- **The artist sets the shot.** Animators compose the opening frame (characters, props, background, staging), so the composition and posing decisions stay with your team. The engine handles the in-between motion.
- **Start-and-end poses (optional).** Beyond a single opening frame, we can let an artist set both a start and an end pose for a shot, so the engine fills the motion between them. This lines up with your existing animatic/posing step.
- **A sign-off rubric.** Before we generate anything for review, your animation director approves a simple scorecard: Character Identity, Motion Quality, Loop Smoothness, Background Stability, Overall Look (scored 1 to 4). Every delivered GIF is scored against it, and the bar to pass is a solid average with no weak spots.
- **You review every batch.** Nothing ships without your team's eyes on it.

---

## Prompting: you write skits, the engine speaks your language

You are professional skit artists, so we build the prompting around *your* workflow instead of asking you to learn a new syntax.

- **Write skits the way you already do.** Screenplay-style beats, exactly like your "Survival" script ("EXT. STREET, Polly waddles next to Pax, about to step into traffic, Pax throws out a flipper to stop her"). That is the input.
- **The engine translates for you.** A step in the pipeline turns your skit into the per-shot instructions the model needs, and fills in the character descriptions automatically. You never write technical prompts.
- **We teach it your words.** The engine is trained using your own terminology (waddle, 2-shot, reverse angle, squash-and-stretch, beat, POV), so your natural descriptions land the way you intend.
- **Your craft stays yours.** Staging, camera framing, and comedic timing are things text can never fully capture, so those stay with your artists through composing the opening frame and setting key poses. The engine handles the in-between motion.

To make this fit your team, we will want a handful of your existing skit scripts (with their final videos) and any internal terminology, so the engine learns to read skits in your voice.

---

## What it will and won't do (honest scope)
- **First deliverable:** short, **seamless looping GIFs** (a few seconds), the format you use on GIPHY. This is the proof point for consistency and production value.
- **Longer shorts come next.** Open AI video can't yet hold 30 seconds in one continuous take without drifting; the standard approach (and ours) is to generate short shots and stitch them, the same way you composite in After Effects today. We start with loops, then build toward 30 second narrative shorts.
- **Assist first, automate later.** Short term this augments your animators (faster in-betweening and variations); longer term it can take on more of the shot.
- **Props can move.** Since you confirmed any prop can animate depending on the skit, we train so props can either hold still or move on cue (e.g. the orca's chomp), not just sit frozen.

---

## Timeline (high level)
| Phase | What happens | Your involvement |
|---|---|---|
| Data prep | Build the training set from your assets | Send assets; confirm the quality rubric |
| Training | Teach the model your characters and style | Review early character tests |
| Pipeline build | Assemble the GIF production flow | First demo + feedback |
| Production + review | Generate, score, and refine the GIF library | Director scores each batch |
| Handoff | Deliver the system + a simple way to run it | Training session for your team |

(Roughly a six-week effort once the data is in hand.)

---

## What we need from you
Everything is listed in the **Data Requirements** doc, but the short version: clean base renders of 15 to 25 skits, characters and backgrounds separated where possible, your props/backgrounds library, Pax and Polly model sheets, the style guide, and a handful of short reference clips shot for training. Plus, for the prompting side, **5 to 10 of your existing skit scripts with their final videos** and any internal terminology, so the engine learns to read skits in your voice. The sooner these land, the sooner we start.

## What success looks like
A model, tried and tested on your own standards, that produces consistent, on-model Pudgy animation, with your director signing off the quality. If the looping GIFs clear your bar, we have a clear path to scale into the Instagram short-form content.
