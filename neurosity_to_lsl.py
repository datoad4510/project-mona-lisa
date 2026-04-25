"""Bridge Neurosity Crown raw EEG to an LSL EEG stream.

Plain-text credentials are intentionally used here per project preference.
Fill DEVICE_ID if needed, then run this script before LabRecorder.
"""

from threading import Event

from neurosity import NeurositySDK
from pylsl import StreamInfo, StreamOutlet, local_clock


# Plain-text config (requested)
EMAIL = "david.adamashvili71@gmail.com"
PASSWORD = "Irakli58"
DEVICE_ID = ""  # Put your Crown device_id here if your account has multiple devices

LSL_STREAM_NAME = "Neurosity_EEG"
LSL_STREAM_TYPE = "EEG"
LSL_SOURCE_ID = "neurosity_crown_raw_v1"


def main():
    if not EMAIL or not PASSWORD:
        raise SystemExit("Set EMAIL and PASSWORD in this file.")

    cfg = {}
    if DEVICE_ID:
        cfg["device_id"] = DEVICE_ID
    neurosity = NeurositySDK(cfg)

    print("Logging in to Neurosity...")
    neurosity.login({"email": EMAIL, "password": PASSWORD})
    info = neurosity.get_info()
    print("Connected device info:", info)

    outlet = {"value": None}
    state = {"sample_index": 0}

    def on_raw(epoch):
        data = epoch.get("data", [])
        if not data:
            return

        n_channels = len(data)
        n_samples = len(data[0]) if data[0] else 0
        if n_samples == 0:
            return

        if outlet["value"] is None:
            stream_info = StreamInfo(
                name=LSL_STREAM_NAME,
                type=LSL_STREAM_TYPE,
                channel_count=n_channels,
                nominal_srate=256.0,  # Crown raw commonly 256 Hz
                channel_format="float32",
                source_id=LSL_SOURCE_ID,
            )
            outlet["value"] = StreamOutlet(stream_info)
            print(f"LSL stream started: {LSL_STREAM_NAME} ({n_channels} channels)")

        base_ts = epoch.get("timestamp")
        # Neurosity timestamp is in ms; convert to seconds if present.
        if base_ts is not None:
            base_ts = float(base_ts) / 1000.0
        else:
            base_ts = local_clock()

        # data shape: channels x samples, LSL expects one sample vector per timestep
        for sample_i in range(n_samples):
            sample_vec = [float(data[ch][sample_i]) for ch in range(n_channels)]
            ts = base_ts + (sample_i / 256.0)
            outlet["value"].push_sample(sample_vec, ts)
            state["sample_index"] += 1

            if state["sample_index"] % 256 == 0:
                print(f"Pushed {state['sample_index']} samples to LSL")

    unsubscribe = neurosity.brainwaves_raw(on_raw)
    stop = Event()

    print("Streaming raw EEG to LSL. Press Ctrl+C to stop.")
    try:
        stop.wait()
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        unsubscribe()


if __name__ == "__main__":
    main()
