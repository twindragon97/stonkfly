"""Only RGB and engineered reinforcement enter the network. No market policy."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .common import annotations
from .visual import VisualMemoryBrain


class Decoder:
    def __init__(self, ids, annotation, threshold):
        types = annotation.type.fillna("")
        sides = annotation.somaSide.fillna("")
        self.left = np.flatnonzero(types.eq("DNp20") & sides.eq("L"))
        self.right = np.flatnonzero(types.eq("DNp20") & sides.eq("R"))
        self.gate = np.flatnonzero(types.eq("DNpe017"))
        if not len(self.left) or not len(self.right) or not len(self.gate):
            raise RuntimeError("Missing annotated BCI outputs")
        self.threshold = threshold
        self.identities = {
            k: [str(ids[i]) for i in getattr(self, k)]
            for k in ["left", "right", "gate"]
        }

    def decode(self, counts, seconds):
        # Mean rates prevent side population size from creating a built-in bias.
        left = float(np.mean(counts[self.left]) / seconds)
        right = float(np.mean(counts[self.right]) / seconds)
        difference = right - left
        gate = int(counts[self.gate].sum())
        side = (
            "HOLD"
            if not gate or abs(difference) < self.threshold
            else "BUY"
            if difference > 0
            else "SELL"
        )
        return {
            "side": side,
            "left_hz": left,
            "right_hz": right,
            "difference_hz": difference,
            "gate_spikes": gate,
            "cell_ids": self.identities,
        }


class FlyController:
    def __init__(self, settings):
        self.s = settings
        self.brain = VisualMemoryBrain()
        self.brain.weights_frozen = not settings.learning
        self.decoder = Decoder(
            self.brain.ids, annotations(self.brain.ids), settings.decoder_threshold_hz
        )

    def observe(self, rgb, reinforcement):
        if reinforcement not in ("none", "reward", "aversive"):
            raise ValueError("Unknown reinforcement")
        b = self.brain
        counts = np.zeros(b.n, dtype=np.int32)
        wall = 0.0
        remaining = round(self.s.neural_ms / b.dt)
        pulse = round(self.s.pulse_ms / b.dt) if reinforcement != "none" else 0
        delivered = 0
        while remaining:
            n = min(remaining, round(self.s.neural_bin_ms / b.dt))
            if pulse:
                n = min(n, pulse)
            stimulus = (
                (b.circuit[reinforcement], self.s.pulse_current) if pulse else None
            )
            c, elapsed = b.rgb_step(
                rgb, n * b.dt, learning=self.s.learning, stimulation=stimulus
            )
            counts += c
            wall += elapsed
            remaining -= n
            if pulse:
                delivered += n
                pulse -= n
        b.counts[:] = counts
        return {
            **self.decoder.decode(counts, self.s.neural_ms / 1000),
            "brain_ms": b.sim_ms,
            "compute_seconds": wall,
            "stimulus": reinforcement,
            "stimulus_ms": delivered * b.dt,
            "reward_spikes": int(counts[b.circuit["reward"]].sum()),
            "aversive_spikes": int(counts[b.circuit["aversive"]].sum()),
            "KC_spikes": int(counts[b.circuit["kc"]].sum()),
            "total_spikes": int(counts.sum()),
            "spike_sha256": hashlib.sha256(counts.tobytes()).hexdigest(),
            "input_sha256": hashlib.sha256(np.asarray(rgb).tobytes()).hexdigest(),
            "memory": b.memory(),
        }

    def save(self, path):
        self.brain.checkpoint(path)

    def restore(self, path):
        self.brain.restore(path)

    def memory_signature(self):
        from .common import digest
        b = self.brain
        return {"format": "stonkfly-memory-v1", "configuration": b.configuration_signature(),
                "kernel": b.build["source_sha256"], "eta": b.eta,
                "ids": digest(b.ids), "ptr": digest(b.ptr), "post": digest(b.post),
                "edges": digest(b.circuit["edges"])}

    def save_memory(self, path):
        """Transfer learned efficacies, never balances, activity or order state."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".partial")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, metadata=json.dumps(self.memory_signature()),
                                memory_u=self.brain.memory_u, memory_w=self.brain.memory_w)
        temporary.replace(path)

    def load_memory(self, path):
        from .rule import PARAMETERS
        b = self.brain
        with np.load(path, allow_pickle=False) as data:
            if json.loads(str(data["metadata"])) != self.memory_signature():
                raise ValueError("Pretrained memory provenance mismatch")
            values = {}
            for key in ("memory_u", "memory_w"):
                value = data[key]
                if (value.shape != getattr(b, key).shape or value.dtype != getattr(b, key).dtype
                    or not np.isfinite(value).all()
                    or np.any(value < PARAMETERS["minimum_fraction"] - 1)
                    or np.any(value > PARAMETERS["maximum_fraction"] - 1)):
                    raise ValueError("Invalid pretrained memory")
                values[key] = value.copy()
        b.reset()
        for key, value in values.items():
            getattr(b, key)[:] = value
        b.weight[b.circuit["edges"]] = b.baseline_plastic * (1 + b.memory_w)
