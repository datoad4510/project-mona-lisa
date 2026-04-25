"""SSVEP calibration: flicker a centered dot at each frequency in turn.

A white dot is shown in the center of the screen and toggled on/off at a
sequence of frequencies, each followed by a blank rest period. LSL markers
are emitted at every state transition so EEG epochs can be cut offline.

Frame-locked square-wave flicker on a fixed-refresh display. Each requested
frequency is snapped to the nearest exact integer frame period
(f = refresh_rate / n_frames_per_cycle).
"""

import argparse
import json
import sys
import uuid

from psychopy import core, event, visual
from pylsl import StreamInfo, StreamOutlet, local_clock


DEFAULT_FREQS = [8.0, 10.0, 12.0, 15.0]
DEFAULT_FLICKER_S = 10.0
DEFAULT_REST_S = 5.0
FALLBACK_REFRESH_HZ = 300.0
DEFAULT_DOT_SIZE_PX = 120


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--frequencies",
        nargs="+",
        type=float,
        default=DEFAULT_FREQS,
        help="Target flicker frequencies in Hz, space-separated.",
    )
    p.add_argument("--flicker-s", type=float, default=DEFAULT_FLICKER_S,
                   help="Seconds of flicker per frequency.")
    p.add_argument("--rest-s", type=float, default=DEFAULT_REST_S,
                   help="Seconds of blank rest between frequencies.")
    p.add_argument(
        "--refresh",
        type=float,
        default=None,
        help=(
            "Monitor refresh rate in Hz. If omitted, PsychoPy's measured value is used. "
            f"If measurement fails, fallback is {FALLBACK_REFRESH_HZ:.0f} Hz."
        ),
    )
    p.add_argument("--dot-size-px", type=int, default=DEFAULT_DOT_SIZE_PX)
    p.add_argument("--windowed", action="store_true",
                   help="Run in a window instead of fullscreen (timing less reliable).")
    return p.parse_args()


def snap_to_refresh(freq, refresh):
    """Return (n_frames_per_cycle, actual_freq) for the closest exact f = R/n."""
    n = max(2, round(refresh / freq))
    return n, refresh / n


def make_marker_outlet():
    info = StreamInfo(
        name="SSVEP_Markers",
        type="Markers",
        channel_count=1,
        nominal_srate=0,
        channel_format="string",
        source_id="ssvep_calib_v1",
    )
    return StreamOutlet(info)


def push_marker(outlet, event_name, **metadata):
    """Push a JSON marker with event name and metadata."""
    ts = local_clock()
    payload = {"event": event_name, "lsl_ts": ts}
    payload.update(metadata)
    outlet.push_sample([json.dumps(payload, separators=(",", ":"))], ts)


def run_flicker(win, dot, n_cycle, n_frames, update_layout):
    """Square-wave flicker for n_frames; dot is on for the first half of each cycle.

    Returns (aborted, dropped_frames).
    """
    half = n_cycle / 2.0
    win.frameIntervals = []
    win.recordFrameIntervals = True

    aborted = False
    for i in range(n_frames):
        update_layout()
        if (i % n_cycle) < half:
            dot.draw()
        win.flip()
        if event.getKeys(keyList=["escape"]):
            aborted = True
            break

    win.recordFrameIntervals = False
    intervals = list(win.frameIntervals)
    if intervals:
        expected = sum(intervals) / len(intervals)
        dropped = sum(1 for fi in intervals if fi > expected * 1.5)
    else:
        dropped = 0
    return aborted, dropped


def run_rest(win, n_frames, update_layout):
    aborted = False
    for _ in range(n_frames):
        update_layout()
        win.flip()
        if event.getKeys(keyList=["escape"]):
            aborted = True
            break
    return aborted


def wait_for_start(win, intro_stim, update_layout):
    """Render intro continuously until SPACE/ESC is pressed."""
    event.clearEvents(eventType="keyboard")
    while True:
        update_layout()
        intro_stim.draw()
        win.flip()
        keys = event.getKeys(keyList=["space", "escape", "return"])
        if "escape" in keys:
            return False
        if "space" in keys or "return" in keys:
            return True


def show_text(win, text_stim, text, n_frames, update_layout):
    """Show a text cue for n_frames; allow ESC to abort."""
    aborted = False
    text_stim.text = text
    for _ in range(n_frames):
        update_layout()
        text_stim.draw()
        win.flip()
        if event.getKeys(keyList=["escape"]):
            aborted = True
            break
    return aborted


def recenter_stimuli(win, dot, intro_stim, cue_stim):
    """Keep all stimuli centered after window size/fullscreen changes."""
    dot.pos = (0, 0)
    intro_stim.pos = (0, 0)
    cue_stim.pos = (0, 0)

    width = float(win.size[0]) if win.size is not None else 1024.0
    intro_stim.wrapWidth = max(500.0, width * 0.85)
    cue_stim.wrapWidth = max(500.0, width * 0.85)


def make_layout_updater(win, dot, intro_stim, cue_stim):
    """Return a function that keeps layout centered whenever window size changes."""
    last_size = [None, None]

    def update():
        size = tuple(int(v) for v in win.size)
        if size != tuple(last_size):
            # Ensure projection/viewport match new size and then recenter.
            win.onResize(*size)
            recenter_stimuli(win, dot, intro_stim, cue_stim)
            last_size[0], last_size[1] = size

    return update


def enable_windowed_native_controls(win):
    """Enable native window controls (including macOS fullscreen button)."""
    handle = getattr(win, "winHandle", None)
    if handle is None:
        return
    if not hasattr(handle, "_resizable"):
        return

    # On macOS, avoid recreating the OpenGL window; just enable resizable/fullscreen
    # style bits on the native NSWindow. Recreate can cause black output on resize.
    if sys.platform == "darwin":
        nswindow = getattr(handle, "_nswindow", None)
        if nswindow is None:
            return
        try:
            from AppKit import (
                NSWindowCollectionBehaviorFullScreenPrimary,
                NSWindowStyleMaskResizable,
            )

            style_mask = int(nswindow.styleMask())
            nswindow.setStyleMask_(style_mask | int(NSWindowStyleMaskResizable))

            behavior = int(nswindow.collectionBehavior())
            nswindow.setCollectionBehavior_(
                behavior | int(NSWindowCollectionBehaviorFullScreenPrimary)
            )
            handle._resizable = True
        except Exception as exc:
            print(f"WARNING: could not enable macOS native window controls: {exc}")
        return

    if handle._resizable:
        return
    if not hasattr(handle, "_recreate"):
        return
    try:
        handle._resizable = True
        handle._recreate(["resizable"])
    except Exception as exc:
        print(f"WARNING: could not enable native resizable window controls: {exc}")


def main():
    if sys.version_info < (3, 9) or sys.version_info >= (3, 12):
        raise SystemExit(
            "This script requires Python >=3.9 and <3.12 for PsychoPy support. "
            "Use python3.11 to create the virtual environment."
        )

    args = parse_args()
    session_id = uuid.uuid4().hex[:8]

    win = visual.Window(
        size=(1024, 768),
        color="black",
        units="pix",
        fullscr=not args.windowed,
        allowGUI=args.windowed,
        waitBlanking=True,
    )
    if args.windowed:
        enable_windowed_native_controls(win)
        win.mouseVisible = True

    measured = win.getActualFrameRate(nIdentical=10, nMaxFrames=120, nWarmUpFrames=10, threshold=1)
    # If --refresh is omitted (or set to 0), use auto-detected refresh.
    refresh_hz = None if args.refresh in (None, 0.0) else float(args.refresh)
    if measured is None:
        if refresh_hz is None:
            refresh_hz = FALLBACK_REFRESH_HZ
        print(f"WARNING: could not measure refresh; using {refresh_hz} Hz")
    else:
        if refresh_hz is None:
            refresh_hz = float(measured)
            print(f"Measured refresh: {measured:.3f} Hz (using measured value)")
        else:
            print(f"Measured refresh: {measured:.3f} Hz (configured {refresh_hz} Hz)")
            if abs(measured - refresh_hz) > 1.0:
                print("WARNING: measured refresh differs from configured value; flicker frequencies will be off.")

    # Normalize for downstream code/markers.
    args.refresh = float(refresh_hz)

    schedule = [(f, *snap_to_refresh(f, args.refresh)) for f in args.frequencies]
    print("Frequency schedule:")
    for requested, n, actual in schedule:
        print(f"  requested {requested:6.2f} Hz -> {n} frames/cycle -> actual {actual:7.4f} Hz")

    dot = visual.Circle(
        win,
        radius=args.dot_size_px / 2,
        fillColor="white",
        lineColor="white",
        units="pix",
    )
    intro = visual.TextStim(
        win,
        text=(
            "SSVEP calibration\n\n"
            "Click the experiment window, then press SPACE.\n"
            "Stare at the center dot during flicker.\n\n"
            "SPACE/ENTER to start  -  ESC to abort"
        ),
        color="white",
        height=28,
        wrapWidth=900,
    )
    cue = visual.TextStim(
        win,
        text="",
        color="white",
        height=36,
        wrapWidth=900,
    )
    recenter_stimuli(win, dot, intro, cue)
    update_layout = make_layout_updater(win, dot, intro, cue)
    win.backend.onResizeCallback = lambda w, _x, _y: recenter_stimuli(w, dot, intro, cue)

    outlet = make_marker_outlet()

    if not wait_for_start(win, intro, update_layout):
        win.close()
        core.quit()
        return

    win.flip()
    core.wait(0.2)

    push_marker(
        outlet,
        "session_start",
        session_id=session_id,
        requested_frequencies_hz=list(args.frequencies),
        flicker_s=args.flicker_s,
        rest_s=args.rest_s,
        refresh_hz=args.refresh,
        windowed=bool(args.windowed),
    )

    aborted = False
    total_dropped = 0

    for trial_index, (requested, n_cycle, actual) in enumerate(schedule, start=1):
        n_flicker_frames = int(round(args.flicker_s * args.refresh))
        n_rest_frames = int(round(args.rest_s * args.refresh))
        n_cue_frames = int(round(1.0 * args.refresh))

        push_marker(
            outlet,
            "cue_on",
            session_id=session_id,
            trial_index=trial_index,
            requested_freq_hz=requested,
            actual_freq_hz=actual,
            n_frames_per_cycle=n_cycle,
        )
        aborted = show_text(win, cue, f"Next: {actual:.2f} Hz", n_cue_frames, update_layout)
        push_marker(
            outlet,
            "cue_off",
            session_id=session_id,
            trial_index=trial_index,
        )
        if aborted:
            break

        push_marker(
            outlet,
            "ball_on",
            session_id=session_id,
            trial_index=trial_index,
            requested_freq_hz=requested,
            actual_freq_hz=actual,
            n_frames_per_cycle=n_cycle,
            planned_flicker_frames=n_flicker_frames,
        )
        aborted, dropped = run_flicker(win, dot, n_cycle, n_flicker_frames, update_layout)
        push_marker(
            outlet,
            "ball_off",
            session_id=session_id,
            trial_index=trial_index,
            requested_freq_hz=requested,
            actual_freq_hz=actual,
            late_frames=dropped,
        )
        total_dropped += dropped
        print(f"  {actual:6.2f} Hz: {dropped} late frames")
        if aborted:
            break

        push_marker(
            outlet,
            "rest_on",
            session_id=session_id,
            trial_index=trial_index,
            rest_s=args.rest_s,
            planned_rest_frames=n_rest_frames,
        )
        aborted = run_rest(win, n_rest_frames, update_layout)
        push_marker(
            outlet,
            "rest_off",
            session_id=session_id,
            trial_index=trial_index,
        )
        if aborted:
            break

    push_marker(
        outlet,
        "session_end",
        session_id=session_id,
        aborted=bool(aborted),
        total_late_frames=total_dropped,
    )

    win.close()
    print(f"Total late frames: {total_dropped}")
    core.quit()


if __name__ == "__main__":
    main()
