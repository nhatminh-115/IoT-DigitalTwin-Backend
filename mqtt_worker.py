"""
MQTT Worker: HiveMQ Cloud → Supabase
Topic: esp/+ | Payload: {"t1": 30.2, "h1": 66.5, "co2": 400, "tvoc": 0, ...}

Usage:
    pip install paho-mqtt supabase python-dotenv
    python mqtt_worker.py
"""

import json
import logging
import os
import queue
import signal
import threading
import time
from datetime import datetime, timezone

import paho.mqtt.client as mqtt
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HIVEMQ_HOST     = os.environ["HIVEMQ_HOST"]
HIVEMQ_PORT     = int(os.environ.get("HIVEMQ_PORT", 8883))
HIVEMQ_USER     = os.environ["HIVEMQ_USER"]
HIVEMQ_PASS     = os.environ["HIVEMQ_PASS"]
HIVEMQ_CLIENTID = os.environ.get("HIVEMQ_CLIENTID", "supabase-worker-01")

SUPABASE_URL    = os.environ["SUPABASE_URL"]
SUPABASE_KEY    = os.environ["SUPABASE_SERVICE_KEY"]

TABLE_NAME      = "env_readings"
FLUSH_INTERVAL  = int(os.environ.get("FLUSH_INTERVAL_SEC", 30))
FLUSH_BATCH     = int(os.environ.get("FLUSH_BATCH_SIZE", 100))
QUEUE_MAXSIZE   = int(os.environ.get("QUEUE_MAXSIZE", 10000))

DEVICE_TO_NODE = {
    "esp01": "M1",  "esp04": "M4",  "esp06": "M6",  "esp07": "M7",
    "esp08": "M8",  "esp09": "M9",  "esp10": "M10", "esp11": "M11",
}

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
record_queue: queue.Queue = queue.Queue(maxsize=QUEUE_MAXSIZE)
shutdown_event = threading.Event()

# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_message(topic: str, payload_bytes: bytes) -> dict | None:
    device = topic.split("/")[-1]
    node_id = DEVICE_TO_NODE.get(device)
    if node_id is None:
        log.debug(f"Unknown device '{device}', skipping")
        return None

    if not payload_bytes:
        return None

    try:
        data = json.loads(payload_bytes.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        log.warning(f"Bad JSON from {device}: {e}")
        return None

    def to_float(v):
        try: return float(v)
        except (TypeError, ValueError): return None

    def to_int(v):
        try: return int(float(v))
        except (TypeError, ValueError): return None

    return {
        "ts":       datetime.now(timezone.utc).isoformat(),
        "node_id":  node_id,
        "temp":     to_float(data.get("t1")),
        "humidity": to_float(data.get("h1")),
        "co2":      to_int(data.get("co2")),
        "tvoc":     to_int(data.get("tvoc")),
    }

# ---------------------------------------------------------------------------
# MQTT callbacks
# ---------------------------------------------------------------------------

def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        log.info("Connected to HiveMQ Cloud")
        client.subscribe("esp/+", qos=1)
        log.info("Subscribed to: esp/+")
    else:
        log.error(f"MQTT connect failed, rc={rc}")

def on_disconnect(client, userdata, rc, properties=None):
    if rc != 0:
        log.warning(f"Unexpected disconnect rc={rc}, reconnecting...")

def on_message(client, userdata, msg):
    record = parse_message(msg.topic, msg.payload)
    if record:
        try:
            record_queue.put_nowait(record)
        except queue.Full:
            log.warning("record_queue full - dropping message from %s", msg.topic)

# ---------------------------------------------------------------------------
# Flush worker
# ---------------------------------------------------------------------------

def flush_worker(supabase):
    log.info(f"Flush worker started (interval={FLUSH_INTERVAL}s, batch={FLUSH_BATCH})")
    last_flush = time.monotonic()

    while not shutdown_event.is_set():
        elapsed = time.monotonic() - last_flush
        if elapsed >= FLUSH_INTERVAL or record_queue.qsize() >= FLUSH_BATCH:
            batch = []
            while len(batch) < FLUSH_BATCH:
                try:
                    batch.append(record_queue.get_nowait())
                except queue.Empty:
                    break

            if batch:
                try:
                    supabase.table(TABLE_NAME).upsert(batch, on_conflict="ts,node_id").execute()
                    log.info(f"Flushed {len(batch)} records")
                except Exception as e:
                    log.error(f"Supabase upsert failed: {e} — re-queuing")
                    for r in batch:
                        try:
                            record_queue.put_nowait(r)
                        except queue.Full:
                            log.warning(
                                "record_queue full during re-queue - dropping record node=%s ts=%s",
                                r.get("node_id"),
                                r.get("ts"),
                            )

            last_flush = time.monotonic()

        time.sleep(1)

    # Drain on shutdown
    remaining = []
    while True:
        try: remaining.append(record_queue.get_nowait())
        except queue.Empty: break
    if remaining:
        try:
            supabase.table(TABLE_NAME).upsert(remaining, on_conflict="ts,node_id").execute()
            log.info(f"Final flush: {len(remaining)} records")
        except Exception as e:
            log.error(f"Final flush failed: {e}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    def handle_signal(sig, frame):
        log.info("Shutting down...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    log.info("Supabase client ready")

    flusher = threading.Thread(target=flush_worker, args=(supabase,), daemon=True)
    flusher.start()

    client = mqtt.Client(
        client_id=HIVEMQ_CLIENTID,
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
    )
    client.username_pw_set(HIVEMQ_USER, HIVEMQ_PASS)
    client.tls_set()
    client.on_connect    = on_connect
    client.on_disconnect = on_disconnect
    client.on_message    = on_message

    log.info(f"Connecting to {HIVEMQ_HOST}:{HIVEMQ_PORT}...")
    client.connect(HIVEMQ_HOST, HIVEMQ_PORT, keepalive=60)
    client.loop_start()

    while not shutdown_event.is_set():
        time.sleep(1)

    client.loop_stop()
    client.disconnect()
    flusher.join(timeout=10)
    log.info("Worker stopped.")

if __name__ == "__main__":
    main()
