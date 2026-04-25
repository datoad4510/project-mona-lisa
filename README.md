# SSVEP Calibration

Simple visual stimulator for measuring brain response to flickering dots.
A white dot is shown in the center of the screen and toggled on/off at a
sequence of frequencies, each followed by a blank rest period. LSL markers
are emitted at every state transition so EEG epochs can be cut offline.

## Install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

PsychoPy can be heavy on macOS. If `pip install` fails, see the
[PsychoPy install docs](https://www.psychopy.org/download.html#pip-install).
Use Python `>=3.9` and `<3.12` (for example `python3.11`).

## Run

```bash
python3.11 main.py
```

Defaults: 10, 12, 15, 20 Hz; 10 s flicker each; 5 s rest between; 60 Hz monitor; fullscreen.

Common flags:

```bash
python3.11 main.py --frequencies 10 12 15 20 --flicker-s 10 --rest-s 5 --refresh 60
python3.11 main.py --windowed                 # run in a window (timing less reliable)
```

Click the experiment window first, then press `SPACE` (or `ENTER`) to start.
Press `ESC` at any time to abort.
In `--windowed` mode, native window controls are enabled so the top macOS
green button can be used to enter fullscreen.

## Frequency snapping

Each requested frequency is snapped to the nearest exact integer frame
period for the configured refresh rate (`f = R / n`). The actual displayed
frequency is printed at startup and embedded in the `stim_on` marker.

On a 60 Hz display: 10, 12, 15, 20 Hz are exact. 8 Hz snaps to 7.5 Hz
(60 / 8 frames). Use 8.57 (60 / 7) if you want something in that range.

## Recording the EEG

This script only emits LSL markers; it does **not** record the EEG itself.
To capture EEG + markers together:

1. Start EEG bridge:
   ```bash
   python3.11 neurosity_to_lsl.py
   ```
2. Open [LabRecorder](https://github.com/labstreaminglayer/App-LabRecorder)
   and select both `Neurosity_EEG` and `SSVEP_Markers`.
3. Press **Start** in LabRecorder.
4. Run `python3.11 main.py`.
5. Press **Stop** in LabRecorder when the session ends.

Output is an `.xdf` file you can open with `pyxdf` or MNE-Python.

## Marker schema

Markers are JSON strings on stream `SSVEP_Markers` (LSL type `Markers`).
Each marker has:

- `event` (string)
- `lsl_ts` (local LSL timestamp when marker was emitted)
- metadata fields (session/trial/frequency/etc.)

Core events:

- `session_start`
- `cue_on` / `cue_off`
- `ball_on` / `ball_off`  <- ball visible / ball not visible
- `rest_on` / `rest_off`
- `session_end`

Example `ball_on` marker payload:

```json
{
  "event": "ball_on",
  "lsl_ts": 12345.678,
  "session_id": "a1b2c3d4",
  "trial_index": 2,
  "requested_freq_hz": 12.0,
  "actual_freq_hz": 12.0,
  "n_frames_per_cycle": 5,
  "planned_flicker_frames": 600
}
```

## Marker stream test

You can inspect marker events live in another terminal:

```bash
python3.11 print_markers.py
```

## Neurosity plain-text config

`neurosity_to_lsl.py` uses plain-text constants at the top:

- `EMAIL`
- `PASSWORD`
- `DEVICE_ID` (optional if you only have one device)

## Notes on timing

- `--refresh` must match your actual monitor refresh rate or the displayed
  frequencies will be wrong. Many "60 Hz" panels are actually 59.94 Hz; if
  the measured value at startup differs by more than 1 Hz, the script warns.
- Run **fullscreen** for reliable vsync. Windowed mode is for development.
- Late frames per trial are reported in the console and embedded in the
  `stim_off` marker; a clean trial should show 0.
