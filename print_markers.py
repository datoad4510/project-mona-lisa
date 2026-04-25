"""Listen to SSVEP marker stream and print events."""

import json

from pylsl import StreamInlet, resolve_stream


def main():
    print("looking for SSVEP_Markers stream...")
    streams = resolve_stream("name", "SSVEP_Markers")
    inlet = StreamInlet(streams[0])
    print("connected. waiting for markers...")

    while True:
        sample, ts = inlet.pull_sample()
        raw = sample[0] if sample else ""
        try:
            payload = json.loads(raw)
            event = payload.get("event", "unknown")
            print(f"{ts:.6f}  {event}  {payload}")
        except json.JSONDecodeError:
            print(f"{ts:.6f}  raw={raw}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Ending marker listener.")
