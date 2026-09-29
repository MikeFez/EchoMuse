import wave

import pytest

import em_db as db
import em_wake_samples as samples


def _pcm(frames=1, byte=1):
    return bytes([byte, 0]) * (samples.SAMPLE_RATE * 80 // 1000) * frames


def test_sample_files_are_wav_and_device_scoped(tmp_path):
    path = str(tmp_path / "db.sqlite")
    name = samples.filename("dev1")
    assert name and samples.save("dev1", name, _pcm(), db_path=path)
    saved = samples.resolve("dev1", name, db_path=path)
    assert saved and saved.is_file()
    assert samples.resolve("dev2", name, db_path=path) is None
    with wave.open(str(saved), "rb") as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 16000)
        assert wav.getnframes() == samples.SAMPLE_RATE * 80 // 1000


def test_capture_includes_preroll_and_postroll_and_coalesces_score_run():
    capture = samples.WakeCapture()
    for i in range(10):
        capture.feed_audio(_pcm(byte=i), now=i * .08)
    capture.consider(enabled=True, minimum=.2, score=.25, threshold=.5,
                     model="hey_vanessa", now=.8)
    # A sustained score run updates one candidate, rather than opening a new
    # clip each scoring frame.
    capture.consider(enabled=True, minimum=.2, score=.3, threshold=.5,
                     model="hey_vanessa", now=.9)
    assert capture.active["kind"] == "candidate"
    assert capture.active["score"] == .3
    assert capture.feed_audio(_pcm(), now=2.3) is not None
    assert capture.active is None


def test_trigger_promotes_candidate_and_floor_rearms_after_score_drops():
    capture = samples.WakeCapture()
    capture.feed_audio(_pcm(), now=0)
    capture.consider(enabled=True, minimum=.2, score=.21, threshold=.6,
                     model="m", now=.1)
    capture.consider(enabled=True, minimum=.2, score=.7, threshold=.6,
                     model="m", trigger_source="controller", now=.2)
    assert capture.active["kind"] == "trigger"
    assert capture.active["trigger_source"] == "controller"
    capture.feed_audio(_pcm(), now=2)
    capture.consider(enabled=True, minimum=.2, score=.01, threshold=.6,
                     model="m", now=2)
    capture.consider(enabled=True, minimum=.2, score=.22, threshold=.6,
                     model="m", now=2.1)
    assert capture.active is not None


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    db.init(str(tmp_path / "samples.db"))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "samples.db"))
    db.register_new_device("dev1", "127.0.0.1", "v1")
    yield
    if db._conn is not None:
        db._conn.close()
        db._conn = None


def test_sample_labels_are_scoped_and_validated(fresh_db):
    name = samples.filename("dev1")
    db._conn.execute(
        "INSERT INTO wake_samples (device_id, ts, audio_file) VALUES (?, ?, ?)",
        ("dev1", 1, name),
    )
    db._conn.commit()
    row = db.get_wake_samples("dev1")[0]
    assert db.set_wake_sample_label("dev1", row["id"], "not_wake")
    assert not db.set_wake_sample_label("dev1", row["id"], "bad")
    assert not db.set_wake_sample_label("other", row["id"], "wake")
    assert db.get_wake_samples("dev1")[0]["label"] == "not_wake"


def test_retention_is_bounded_per_device(fresh_db):
    for i in range(samples.KEEP_PER_DEVICE + 3):
        db.insert_wake_sample("dev1", {
            "ts": i, "audio_file": samples.filename("dev1"), "kind": "candidate",
            "score": .3, "threshold": .5,
        })
    rows = db.get_wake_samples("dev1", 100)
    assert len(rows) == samples.KEEP_PER_DEVICE
    assert rows[0]["ts"] == samples.KEEP_PER_DEVICE + 2


def test_deleting_device_removes_sample_rows_and_audio(fresh_db):
    name = samples.filename("dev1")
    assert samples.save("dev1", name, _pcm())
    db.insert_wake_sample("dev1", {"ts": 1, "audio_file": name, "kind": "candidate"})
    path = samples.resolve("dev1", name)
    assert path and path.is_file()
    db.delete_device("dev1")
    assert db.get_wake_samples("dev1") == []
    assert not path.exists()
