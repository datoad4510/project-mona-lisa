# BCI Observability Improvements

The current logging only emits a heartbeat line every 5 seconds from the headset thread
(`chunks=N  samples=N  rate=Hz`). Remove that and replace it with the four targeted
additions below. Each targets a specific point in the pipeline where the system is
currently silent.

---

## 1. Save every training session to disk

Add to `_on_training_complete` in `ui/app.py`, before calling `self._classifier.train()`:

```python
import pathlib, datetime

ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
out = pathlib.Path("sessions") / f"session_{ts}.npz"
out.parent.mkdir(exist_ok=True)
np.savez(out, epochs=np.array(epochs), labels=np.array(labels))
logging.info("Session saved → %s", out)
```

Inspect a saved session offline:

```python
import numpy as np, matplotlib.pyplot as plt

d = np.load("sessions/session_XXXXXXXX.npz")
epochs, labels = d["epochs"], d["labels"]  # (30, 8, 500) and (30,)

# Average epoch per colour on PO7 (channel 5)
for c in range(3):
    avg = epochs[labels == c].mean(axis=0)
    plt.plot(avg[5], label=f"colour {c}")
plt.legend(); plt.show()
```

---

## 2. Per-trial signal quality log

Add to `_show_isi` in `ui/training_screen.py`, replacing the silent drop:

```python
color_name = config.TRAINING_COLORS[color_idx]["name"]

if len(self._epoch_buf) < config.COLOR_EPOCH_SAMPLES:
    logging.warning(
        "Trial %d (%s): DROPPED — only %d/%d samples buffered",
        self._trial_idx, color_name,
        len(self._epoch_buf), config.COLOR_EPOCH_SAMPLES,
    )
    # continue to ISI without saving
else:
    arr = np.array(list(self._epoch_buf)[-config.COLOR_EPOCH_SAMPLES:]).T
    ch_var = np.var(arr, axis=1)
    for name, var in zip(config.CHANNEL_NAMES, ch_var):
        if var < 0.1:
            logging.warning("Trial %d (%s): channel %s FLAT (var=%.3f)",
                            self._trial_idx, color_name, name, var)
        elif var > 5000:
            logging.warning("Trial %d (%s): channel %s SATURATED (var=%.1f)",
                            self._trial_idx, color_name, name, var)
    logging.info("Trial %d (%s): saved — var min=%.1f max=%.1f",
                 self._trial_idx, color_name, ch_var.min(), ch_var.max())
    self._epochs.append(arr.astype(np.float32))
    self._labels.append(color_idx)
```

---

## 3. Training summary after fitting

Add to `ColorClassifier.train()` in `bci/color_classifier.py`, after computing LOO scores:

```python
logging.info("=== Training report ===")
logging.info("  Epochs per class : %s",
             {int(c): int((y == c).sum()) for c in np.unique(y)})
logging.info("  LOO-CV accuracy  : %.1f%%", scores.mean() * 100)
logging.info("  Feature shape    : %s", X.shape)

# Which features separate the classes best?
class_means = np.array([X[y == c].mean(axis=0) for c in np.unique(y)])
between_var = np.var(class_means, axis=0)
top5 = np.argsort(between_var)[-5:][::-1]
band_names = list(config.BANDS.keys())
feat_labels = [
    f"{config.CHANNEL_NAMES[i // 6]}/{band_names[i % 6]}"
    for i in top5
]
logging.info("  Top-5 discriminating features: %s", feat_labels)
```

If LOO-CV is at or near 33% (chance level for 3 classes), the signal preprocessing
needs fixing before anything else matters.

---

## 4. Inference confidence in the status bar

Expose the full probability vector from `ColorClassifier.predict()` in
`bci/color_classifier.py`:

```python
# Change the return type to include all class probabilities
return best_idx, float(proba[best_idx]), proba.tolist()
```

Update `_update_color_from_classifier` in `ui/app.py` to show confidence:

```python
def _update_color_from_classifier(self) -> None:
    result = self._classifier.predict()
    if result is None:
        return
    color_idx, confidence, all_probas = result
    self._current_color_idx = color_idx
    colors = config.TRAINING_COLORS[:config.N_TRAINING_COLORS]
    if color_idx < len(colors):
        color_info = colors[color_idx]
        self._color_display.setStyleSheet(
            f"background: rgb({color_info['rgb'][0]},{color_info['rgb'][1]},"
            f"{color_info['rgb'][2]}); border-radius: 6px; color: white; font-size: 13px;"
        )
        self._color_display.setText(color_info["name"])
        self._color_confidence.setValue(int(confidence * 100))

    # Log full distribution every ~2 s (every 20 ticks at 100 ms)
    self._conf_log_counter = getattr(self, "_conf_log_counter", 0) + 1
    if self._conf_log_counter % 20 == 0:
        dist = "  ".join(
            f"{c['name']}={p:.0%}"
            for c, p in zip(colors, all_probas)
        )
        logging.debug("Inference: %s", dist)
```

If confidence is consistently ≈33% across all three colours the classifier is
guessing — a direct indicator that the input signal is not informative.
