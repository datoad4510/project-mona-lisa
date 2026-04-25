# BCI Colour Training — Evaluation

> Unicorn Hybrid Black · 3 colours · LDA classifier

---

## Does it do what it's supposed to?

The structure is correct: the training screen shows a colour, collects EEG during the stimulus, associates the epoch to a label, and trains an LDA classifier. The inference loop also works architecturally. However, data quality issues mean what reaches the model is dirty enough to undermine the whole exercise.

---

## Critical issues (will corrupt data)

### 1. SSVEP flicker runs during training

The flickering circles are live the entire time the colour patches are shown. The 8, 10, 12 Hz flickers drive visual-evoked responses that land inside the alpha and beta bands — the same bands used as colour features. This is the most serious contamination source.

**Fix:** disable all flicker overlays for the entire training phase.

### 2. No notch filter

50 Hz mains noise contaminates the gamma band on every epoch. `butter_bandpass` is already written in `signal_processing.py` — it is just never called.

**Fix:** apply a 50 Hz notch filter (and a 0.5 Hz high-pass) before feature extraction.

### 3. No detrending / DC removal

Electrode drift inflates delta power and log-variance, adding noise unrelated to colour.

**Fix:** apply `scipy.signal.detrend(epoch, axis=1)` before bandpower computation.

---

## High priority issues

### 4. Onset transient included in epoch

EEG collection starts the instant the colour patch appears (`_collecting = True` in `_show_stimulus`). The first ~300–500 ms of every epoch is a visual onset response (VEP / P300-like burst), not the sustained colour-imagery signal the model needs to learn. This transient looks the same for every colour, so it actively confuses the classifier.

**Fix:** discard the first 500 ms of every epoch — skip the first 125 samples at 250 Hz. Shift the useful window to 500 ms – 2500 ms after onset.

### 5. Train window (2 s) ≠ inference window (3 s)

`COLOR_EPOCH_SEC = 2.0` s for training, `RECALL_DURATION_SEC = 3.0` s for inference. The Welch PSD statistics differ between window lengths, introducing a train/inference domain shift.

**Fix:** make both windows identical — either both 2 s or both 3 s.

### 6. Silent trial drops

If the EEG buffer holds fewer than `COLOR_EPOCH_SAMPLES` at ISI (slow machine, USB hiccup), the trial is silently discarded. If enough trials drop the app transitions to `READY` with an untrained classifier and no warning to the user.

**Fix:** log a warning on every dropped trial. Before emitting `training_complete`, abort if fewer than `N_CLASSES × 2` epochs were collected.

---

## Medium priority issues

### 7. Epoch window can slip into grey-screen

The epoch is saved at the start of the ISI. If the Qt event loop delays `_show_isi()` by even a few frames, the saved window slides into neutral grey-screen territory rather than the colour response.

**Fix:** snapshot the buffer immediately when the stimulus timer fires, before the repaint.

### 8. LDA with 30 samples and 48 features

With 30 training epochs (10 per colour × 3 colours) and 48 features (8 channels × 6), the feature matrix is underdetermined. LOO-CV accuracy will be inflated and will not reflect real generalisation.

**Fix:** reduce to 4 channels (C3, C4, PO7, PO8) × alpha + beta = 8 features. Use `shrinkage='auto'` in LDA (`solver='eigen'`). Consider increasing to 15–20 trials per colour.

### 9. 8 Hz SSVEP overlaps the alpha band

The 8 Hz SSVEP frequency sits at the lower edge of the alpha band (8–13 Hz in `config.py`). Any flicker leakage into training corrupts the strongest colour-imagery feature.

**Fix:** shift SSVEP frequencies to 15, 20, 25 Hz to keep all flicker harmonics above the beta band.

---

## Low priority issues

### 10. Bandpower normalisation denominator mismatch

Relative power is computed as `bp / np.var(signal)`. `np.var` is a time-domain statistic while `bp` is the integral of a Welch PSD — they are not the same quantity, so relative powers do not sum to 1.0 across bands.

**Fix:** normalise by the sum of PSD power in 0.5–45 Hz instead of `np.var()`.

### 11. `butter_bandpass` is dead code

`signal_processing.py` defines and exports `butter_bandpass` but nothing in the pipeline calls it.

**Fix:** wire it in as a broadband pre-filter (0.5–45 Hz) applied before band-power estimation, or remove it to avoid confusion.

---

## Recommended preprocessing order

Apply inside `extract_features()` before any band-power computation:

| Step | Function | Parameters |
|------|----------|------------|
| 1. Detrend | `scipy.signal.detrend(epoch, axis=1)` | `type='linear'` |
| 2. High-pass | `butter_bandpass` (already in `signal_processing.py`) | 0.5 Hz cutoff |
| 3. Notch | `iirnotch + sosfiltfilt` | 50 Hz, Q=30 |
| 4. Trim onset | `epoch[:, 125:]` | Drop first 500 ms (125 samples at 250 Hz) |
| 5. Band power | existing `bandpower()` — unchanged | δ θ α β γ |
| 6. Normalise | sum of PSD power in 0.5–45 Hz | replace `np.var()` denominator |

---

## Classifier recommendations

- **Window alignment** — set `COLOR_EPOCH_SEC = RECALL_DURATION_SEC = 2.0` so training and inference use identically-sized windows.
- **Feature reduction** — 4 channels × 2 bands = 8 features is more appropriate for 30–60 training samples than the current 48.
- **Shrinkage LDA** — use `LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')` to handle small-N robustly.
- **More trials** — aim for 15–20 trials per colour (45–60 total) before trusting LOO-CV accuracy.
- **SSVEP frequency shift** — move to 15, 20, 25 Hz so flicker never aliases into colour feature bands.
