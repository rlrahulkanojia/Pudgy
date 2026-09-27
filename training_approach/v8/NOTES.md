# v8 notes: dataset and the maths behind it

Companion to the [v8 README](README.md) (how to prompt) and the full technical plan,
[`Training_Approach_v8.md`](Training_Approach_v8.md).

## 1. The v8 dataset

**`processed/v8_joint_2096`**: 2,096 clips from 193 client performances, all at 1024×1024
and 24 fps, silent, with the character placed on a flat colour background. A further
26 performances (the `QF2_R` camera angle) are held back for testing only.

| | Performances | Clips | Labels |
|---|---|---|---|
| Movements | 77 | 896 | walking, running, waving, sitting, jumping |
| Expressions | 116 | 1,200 | happy, surprised, angry, confused, crying, neutral, laughing, **blushing (new)** |

- **Balanced on purpose.** Pax 1,028 / Polly 1,068 clips; exactly 524 clips on each of 4
  backgrounds; wide / medium / close-up shots split evenly; 8 camera angles.
- **Several lengths per label** (0.5–2.4 s), so length can be requested and doesn't give away
  the label.
- **Looped versions** (2 s, repeating) of walking, waving, jumping and sitting, made only
  where the cycle joins seamlessly. Running doesn't loop, so it isn't looped.
- **Built from** client deliveries 3, 4 and 5; 288 duplicate re-sent files were detected
  and skipped.

### Issues found in the data

| Issue | Effect | What we did |
|---|---|---|
| **21 client files are missing frames** (all 18 sitting clips, 4 confused, 1 walking, 1 blushing). The video decoder silently fills each gap with a copy of the previous frame | Stutter mid-motion. v7 trained on these without knowing | Clips are cut before the first missing frame; 5 files dropped entirely. **Re-export requested, not yet delivered** |
| **Corrupt files** (torn frames) | Broken frames | Checked on every frame; 1 more sitting file dropped |
| **One Polly blushing clip filed in Pax's folder** | Would have trained a pink penguin as Pax | Character read from the file name, not the folder |
| **Folder named `Pax` instead of `PAX`** | v7 silently skipped it | Names now matched regardless of case |
| **"Pink blush" in a Pax caption** | "Pink" is Polly's colour cue; risks mixing characters | Captioned "rosy" instead |
| **`confused` and `happy` delivered at one short length only** | 48 clips each vs 192 for other expressions, so about ¼ of the training signal | Left as is (decision 26 Sep); watched at every test point |
| **Running doesn't loop** | No smooth repeating run | Not looped |
| **12 two-character clips** (hugging, waving together) | Need a different caption design | Not used in v8 |
| **Still missing:** standing idle, sad, scared, affectionate, turning | Can't be taught | Requested from the client |
| **Held-out angle already seen by v7** | Since v8 starts from v7, that angle is no longer truly new | Only the blushing clips are fully unseen; stated in every result |

## 2. The maths behind the v8 choices

Each v8 choice comes from something we measured in the v7 weights or clips.
Details: [`ANALYSIS_v7.md`](../../docs/training_reports/v7/ANALYSIS_v7.md) and
[`Training_Approach_v8.md`](Training_Approach_v8.md).

| # | What we measured | What it meant | What v8 does |
|---|---|---|---|
| 1 | Both v7 add-ons contain the v2 golden. Loading them together counts it **twice: a 76% error** | Using both v7 files together, as v7 was meant to be used, was broken | Start from their **exact correct sum**, built as one file (error 2.5×10⁻⁸) |
| 2 | The two v7 add-ons overlap (similarity 0.33, up to 0.58 in parts). Combined, they keep only **27% / 43%** of what each learned alone | Adding them makes them interfere | Train **one joint** add-on that learns the shared part once. By step 100 it had recovered most of the loss: **−33% on every test clip** |
| 3 | v7's weight changes were mostly random: only **5% (step 500) to 19% (step 2,500)** was steady learning | More training steps weren't making v7 better | Average 8 clips per update and taper the learning rate. v8 still shows mostly random movement after step 50, so **checkpoint averaging** is planned for the end |
| 4 | v7's test metric (face-SSIM) identified the **random seed 67–81%** of the time and the **label at chance** | v7's checkpoint choice was based on noise | Score clips with CLIP/DINOv2 instead, and pick checkpoints with a **diffusion classifier** (validated: an untrained model scores 1/8, a trained one 7/8) |
| 5 | At high noise the model can only learn a label from **how different the average clip is** per label. For in-place actions that's ≈ 0 | Why motion failed on the high-noise expert | Train both kinds on the low-noise expert only |
| 6 | v7's changes used only an effective **rank of about 9 of 32** per kind | Rank wasn't the limit | Rank 32 for one joint add-on covering both kinds |
| 7 | Loop joins measured as **join size ÷ normal frame step**: walking 0.6–1.0, running 2.3–2.7 | Separates clean loops from broken ones | Loop only below 1.5; checked afterwards: joins are no rougher than normal motion |
| 8 | The footage is **hand-drawn at 24 fps** with smooth easing | Converting to the model's native 16 fps would cause judder or ghosting | Stayed at 24 fps |
| 9 | Telling two checkpoints apart by 10 points needs **~350 clips each** | The old 21-clip tests could never decide | The cheap classifier does the selection; generated clips are the final check |
| 10 | The trainer's resume **restarted its step count at 0** | Would have overwritten checkpoints at the first pause | Patched and verified at step 100 |

**Where it stands (27 Sep, step 240 of 600):** at step 100, all four pre-set targets were met.
Expression recognition on the held-out clips went from 0.63 to 0.81, and loss fell on
every clip. Motion recognition hasn't improved yet; it's the thing to watch at step 300.
