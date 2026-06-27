from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, Mapping, MutableMapping, Sequence

import numpy as np

EmbedFn = Callable[[str], Sequence[float] | np.ndarray]

BASE_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = Path(os.getenv("E8_RUNTIME_DIR", BASE_DIR / "runtime"))
DEFAULT_EMOJI_EMBED_PATH = RUNTIME_DIR / "emoji_embeddings.npz"
DEFAULT_GLYPH_SEMANTIC_PATH = RUNTIME_DIR / "glyph_semantics.npz"
DEFAULT_GLYPH_TO_EMOJI_PATH = RUNTIME_DIR / "glyph_to_emoji.json"

DEFAULT_TEXT_FIELDS = (
    "content",
    "text",
    "label",
    "summary",
    "metaphor",
    "hypothesis",
    "insight",
    "notes",
    "description",
    "ticker",
)

MAX_USAGE_TEXT_CHARS = 2000

EMOJI_CANDIDATES: list[tuple[str, str]] = [
    ("🌱", "sprout new beginning growth"),
    ("🌿", "leaf nature healing organic"),
    ("🌳", "tree stability ecosystem rooted"),
    ("🌲", "pine evergreen resilience alpine"),
    ("🌼", "daisy optimism renewal delicate"),
    ("🌸", "cherry blossom transient beauty bloom"),
    ("🌻", "sunflower radiant focus alignment"),
    ("🌺", "hibiscus tropical flair color"),
    ("🍂", "fallen leaf cycle letting go"),
    ("🍃", "breeze motion whisper air"),
    ("🌧️", "rain cleansing emotion renewal"),
    ("⛈️", "storm intensity disruption breakthrough"),
    ("🌩️", "lightning sudden insight energy"),
    ("🔥", "fire energy transformation danger"),
    ("💧", "droplet clarity emotion cooling"),
    ("🌊", "wave flow rhythm recursion"),
    ("🌀", "spiral vortex recursion swirl"),
    ("🌪️", "tornado upheaval change force"),
    ("🌫️", "fog ambiguity uncertainty mystery"),
    ("🌬️", "wind breath whisper motion"),
    ("🌈", "rainbow spectrum harmony promise"),
    ("❄️", "snowflake uniqueness symmetry calm"),
    ("☀️", "sun vitality illumination warmth"),
    ("🌞", "sun face optimism warmth"),
    ("🌙", "moon intuition night cycles"),
    ("⭐", "star guidance inspiration navigation"),
    ("🌟", "glowing star highlight excellence"),
    ("✨", "sparkles magic novelty special"),
    ("⚡", "electric charge acceleration intuition"),
    ("☄️", "comet trajectory omen momentum"),
    ("🌌", "milky way cosmic wonder depth"),
    ("🌠", "shooting star wish signal path"),
    ("🪐", "ringed planet structure orbit pattern"),
    ("🌐", "globe network global connection"),
    ("🧭", "compass navigation orientation true north"),
    ("🗺️", "map exploration planning terrain"),
    ("🛰️", "satellite telemetry observation relay"),
    ("🚀", "rocket launch ambition escape"),
    ("🛸", "ufo curiosity unknown speculative"),
    ("🛠️", "hammer wrench repair iteration"),
    ("⚙️", "gear mechanism systems tuning"),
    ("🔧", "wrench adjust tweak maintenance"),
    ("🧱", "brick foundation structure incremental"),
    ("🧬", "dna code life blueprint"),
    ("🧠", "brain thinking intelligence mind"),
    ("👁️", "eye awareness perception witness"),
    ("🫀", "anatomical heart authenticity feeling"),
    ("💡", "lightbulb idea insight realization"),
    ("🕳️", "hole void black hole unknown"),
    ("📡", "antenna listening broadcast signal"),
    ("🔭", "telescope focus discovery distant"),
    ("🔬", "microscope detail analysis precision"),
    ("🧪", "test tube experiment hypothesis"),
    ("⚗️", "alembic distill refinement synthesis"),
    ("🧫", "petri dish culture iteration growth"),
    ("📚", "books knowledge learning archive"),
    ("📖", "open book narrative documentation"),
    ("📝", "memo capture notes intent"),
    ("✏️", "pencil drafting sketch planning"),
    ("🧮", "abacus computation structure tradition"),
    ("📊", "bar chart metrics comparison"),
    ("📈", "growth chart improvement trend"),
    ("📉", "decline chart caution correction"),
    ("🎛️", "control knobs tuning modulation"),
    ("🎚️", "slider calibration nuance mixing"),
    ("🧰", "toolbox capability readiness kit"),
    ("🪄", "magic wand orchestration transformation"),
    ("🎯", "bullseye focus accuracy objective"),
    ("🎲", "dice randomness chance exploration"),
    ("♻️", "recycle cycles sustainability renewal"),
    ("🔁", "repeat loop iteration feedback"),
    ("🔂", "single repeat review attention"),
    ("🪞", "mirror reflection self awareness"),
    ("🪟", "window perspective aperture view"),
    ("🧊", "ice clarity preservation pause"),
    ("🪨", "rock stability anchor mass"),
    ("⛰️", "mountain challenge ascent summit"),
    ("🪵", "wood resource craft structure"),
    ("⚓", "anchor grounding stability presence"),
    ("🪢", "knot binding tension relation"),
    ("🧷", "safety pin holding together care"),
    ("🧵", "thread continuity weaving story"),
    ("🪡", "needle precision stitching repair"),
    ("🧶", "yarn creativity patience pattern"),
    ("🪴", "potted plant nurture slow growth"),
    ("🌵", "cactus resilience desert endurance"),
    ("🍄", "mushroom mycelium network emergence"),
    ("🌽", "corn abundance harvest sustenance"),
    ("🍇", "grapes collaboration clusters share"),
    ("🍎", "red apple knowledge temptation health"),
    ("🍋", "lemon zest contrast catalyst"),
    ("🍉", "watermelon refresh summer delight"),
    ("🍑", "peach softness vulnerability sweetness"),
    ("🥝", "kiwi bright tang curiosity"),
    ("🥥", "coconut layers shelter resource"),
    ("🍞", "bread sustenance craft fermentation"),
    ("🧂", "salt seasoning preservation ritual"),
    ("🫘", "beans protein humble fuel"),
    ("🍯", "honey sweetness patience collective"),
    ("🥣", "bowl nourishment gathering sharing"),
    ("🥢", "chopsticks dexterity culture finesse"),
    ("🍽️", "plate readiness hospitality meal"),
    ("🍵", "green tea calm ritual reflection"),
    ("☕", "coffee focus energy warmth"),
    ("🧋", "bubble tea playful texture trend"),
    ("🥤", "straw drink quick refresh"),
    ("🎐", "wind chime signal calm breeze"),
    ("🔔", "bell alert attention ritual"),
    ("📯", "postal horn announcement rally"),
    ("📣", "megaphone broadcast advocacy signal"),
    ("🎙️", "studio microphone voice capture presence"),
    ("🎧", "headphones focus immersion isolation"),
    ("🎵", "music note rhythm harmony melody"),
    ("🥁", "drum cadence heartbeat momentum"),
    ("🎻", "violin nuance emotion resonance"),
    ("🎹", "piano structure pattern intervals"),
    ("🪕", "banjo folk improvisation bounce"),
    ("🎼", "score notation structure arrangement"),
    ("🎨", "palette creativity expression blending"),
    ("🖌️", "paintbrush gesture craft stroke"),
    ("🧩", "puzzle piece fit pattern insight"),
    ("🎭", "theatre masks persona contrast duality"),
    ("🎬", "clapper action narrative framing"),
    ("📽️", "film projector memory playback"),
    ("📷", "camera capture observation framing"),
    ("🕰️", "clock timing cadence patience"),
    ("⏳", "hourglass cycle time threshold"),
    ("⌛", "hourglass done completion ending"),
    ("⏱️", "stopwatch measurement pace timing"),
    ("💾", "floppy memory persistence archive"),
    ("💽", "optical data retro storage"),
    ("💿", "cd spectrum reflection audio"),
    ("🖥️", "desktop computing focus workspace"),
    ("💻", "laptop mobility creation coding"),
    ("⌨️", "keyboard interface input typing"),
    ("🖱️", "mouse pointing precision selection"),
    ("🖇️", "paperclip connection reference linking"),
    ("📎", "paperclip inline attachment join"),
    ("🗂️", "card index organization retrieval"),
    ("🗃️", "card file archive structured"),
    ("📦", "package delivery module container"),
    ("🪙", "coin value exchange economy"),
    ("💰", "money bag capital accumulation"),
    ("💎", "gem clarity rarity precision"),
    ("⚖️", "scales balance ethics judgement"),
    ("🪜", "ladder progress steps altitude"),
    ("🧲", "magnet attraction polarity alignment"),
    ("🧯", "fire extinguisher safety mitigation"),
    ("🚨", "siren alert urgency attention"),
    ("🚧", "construction barrier caution iteration"),
    ("🛣️", "motorway journey throughput direction"),
    ("🛤️", "railway track commitment guidance"),
    ("🤝", "handshake partnership agreement trust"),
    ("🤲", "cupped hands offering receiving care"),
    ("🙏", "hands pressed humility gratitude"),
    ("🫡", "salute respect acknowledgment duty"),
    ("👣", "footprints trail memory progress"),
    ("🧍", "person presence stance awareness"),
    ("🧗", "climber effort ascent persistence"),
    ("🧘", "lotus meditative calm centered"),
    ("🏃", "runner momentum urgency chase"),
    ("🧑‍🚀", "astronaut exploration courage frontier"),
    ("🧑‍🔬", "scientist inquiry lab rigor"),
    ("🧑‍🎨", "artist imagination craft expression"),
    ("🧑‍🚒", "firefighter protection bravery response"),
    ("🧑‍⚖️", "judge discernment fairness evaluation"),
    ("🎓", "graduation learning mastery milestone"),
    ("🏅", "medal achievement recognition honor"),
    ("🏆", "trophy victory completion reward"),
    ("🕊️", "dove peace release grace"),
    ("🦋", "butterfly metamorphosis transformation"),
    ("🐚", "shell resonance listening ocean"),
    ("🐙", "octopus distributed cognition multitask"),
    ("🐬", "dolphin play intelligence signal"),
    ("🦉", "owl wisdom observation night"),
    ("🦢", "swan elegance devotion glide"),
    ("🦜", "parrot communication mimicry color"),
    ("🦈", "shark focus relentless apex"),
    ("🐺", "wolf pack intuition loyalty"),
    ("🦊", "fox adaptability clever stealth"),
    ("🐝", "bee collaboration industrious network"),
    ("🐛", "caterpillar potential patience growth"),
    ("🦂", "scorpion defense vigilance desert"),
    ("🐉", "dragon myth power guardian"),
    ("🦄", "unicorn rarity imagination hopeful"),
    ("🐈‍⬛", "black cat mystery intuition independence"),
    ("🐕", "dog loyalty guidance friend"),
    ("🦮", "guide dog assistance trust support"),
    ("🔒", "lock security commitment privacy"),
    ("🔓", "unlock access reveal permission"),
    ("🗝️", "key secret leverage insight"),
    ("⚪", "white circle openness blank potential"),
    ("⚫", "black circle gravity depth void"),
    ("🔺", "triangle warning focus delta"),
    ("🔻", "inverted triangle release descent"),
    ("🔸", "orange diamond spark emphasis"),
    ("🔹", "blue diamond calm precision"),
    ("🔷", "large blue diamond stability system"),
    ("🔶", "large orange diamond energy flux"),
    ("⬛", "black square solidity constraint boundary"),
    ("⬜", "white square clarity frame plan"),
    ("◀️", "reverse arrow rewind reflection"),
    ("▶️", "play arrow motion forward"),
    ("⏩", "fast forward acceleration skip"),
    ("⏪", "rewind review recollect"),
    ("🔀", "shuffle exploration remapping serendipity"),
    ("🔃", "sync refresh recalibrate cycle"),
    ("📶", "signal bars connectivity bandwidth"),
    ("📳", "vibration alert subtle ping"),
    ("🔇", "mute silence pause focus"),
    ("🔊", "speaker broadcast amplitude share"),
    ("💬", "speech bubble dialogue exchange"),
    ("🗨️", "left speech bubble whisper aside"),
    ("💭", "thought cloud rumination possibility"),
    ("🗯️", "anger bubble decisive urgency"),
    ("📤", "outbox send release publish"),
    ("📥", "inbox intake reception queue"),
    ("🧾", "receipt accountability record audit"),
]


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _normalize(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0 or not math.isfinite(norm):
        return arr
    return arr / norm


def _resize_vector(vec: np.ndarray, target_dim: int) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    if vec.size == target_dim:
        return vec
    if target_dim <= 0:
        raise ValueError("target_dim must be positive")
    if vec.size > target_dim:
        return vec[:target_dim]
    out = np.zeros(target_dim, dtype=np.float32)
    out[: vec.size] = vec
    return out


def _prepare_vector(vec: Sequence[float] | np.ndarray, target_dim: int | None = None) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if target_dim is not None:
        arr = _resize_vector(arr, target_dim)
    return _normalize(arr)


def _log(logger: Callable[[str], None] | None, message: str) -> None:
    if logger is None:
        return
    try:
        logger(message)
    except Exception:
        pass


def build_emoji_embedding_table(
    embed_fn: EmbedFn,
    candidates: Iterable[tuple[str, str]] = EMOJI_CANDIDATES,
    *,
    target_dim: int | None = None,
    logger: Callable[[str], None] | None = None,
) -> dict[str, np.ndarray]:
    """Embed each emoji gloss into the universal embedding space."""

    table: dict[str, np.ndarray] = {}
    for emoji, gloss in candidates:
        gloss = (gloss or "").strip()
        if not gloss:
            continue
        try:
            vec = embed_fn(gloss)
            table[emoji] = _prepare_vector(vec, target_dim=target_dim)
        except Exception as exc:  # pragma: no cover - best effort logging
            _log(logger, f"[emoji_semantics] Failed to embed gloss '{gloss}': {exc}")
            raise
    return table


def save_emoji_embeddings(
    table: Mapping[str, np.ndarray],
    path: str | os.PathLike[str] = DEFAULT_EMOJI_EMBED_PATH,
) -> Path:
    path = Path(path)
    _ensure_parent_dir(path)
    emojis = list(table.keys())
    if emojis:
        vecs = np.stack([np.asarray(table[e], dtype=np.float32) for e in emojis], axis=0)
    else:
        vecs = np.zeros((0, 0), dtype=np.float32)
    np.savez(path, emojis=np.array(emojis, dtype=object), vecs=vecs)
    return path


def load_emoji_embeddings(
    path: str | os.PathLike[str] = DEFAULT_EMOJI_EMBED_PATH,
) -> tuple[list[str], np.ndarray]:
    data = np.load(Path(path), allow_pickle=True)
    emojis = list(data["emojis"])
    vecs = np.asarray(data["vecs"], dtype=np.float32)
    return emojis, vecs


def build_glyph_semantics_from_basis(
    basis_matrix: Sequence[Sequence[float]] | np.ndarray,
    *,
    adapter: Callable[[np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """Option A: project glyph basis vectors through the universal adapter."""

    basis = np.asarray(basis_matrix, dtype=np.float32)
    if basis.ndim != 2:
        raise ValueError("basis_matrix must be 2-D")
    glyph_vecs = []
    for row in basis:
        vec = np.asarray(row, dtype=np.float32)
        if adapter is not None:
            vec = adapter(vec)
        glyph_vecs.append(_normalize(vec))
    return np.stack(glyph_vecs, axis=0) if glyph_vecs else np.zeros((0, 0), dtype=np.float32)


def load_glyph_usage_from_logs(
    paths: Iterable[str | os.PathLike[str]],
    *,
    weight_field: str = "glyph_topk",
    text_fields: Sequence[str] = DEFAULT_TEXT_FIELDS,
) -> dict[int, list[tuple[float, str]]]:
    """Option B: harvest glyph usage from NDJSON logs."""

    usage: dict[int, list[tuple[float, str]]] = defaultdict(list)
    for path in paths:
        p = Path(path)
        if not p.exists() or p.stat().st_size == 0:
            continue
        with p.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                glyphs = record.get(weight_field)
                if not glyphs:
                    continue
                text_parts: list[str] = []
                for key in text_fields:
                    val = record.get(key)
                    if isinstance(val, str):
                        val = val.strip()
                        if val:
                            text_parts.append(val)
                if not text_parts:
                    continue
                text = " ".join(text_parts)
                if len(text) > MAX_USAGE_TEXT_CHARS:
                    text = text[:MAX_USAGE_TEXT_CHARS]
                for item in glyphs:
                    idx: int | None = None
                    weight: float | None = None
                    if isinstance(item, Mapping):
                        raw_idx = item.get("glyph") or item.get("index") or item.get("id")
                        raw_weight = item.get("weight") or item.get("w") or item.get("score") or item.get("value")
                        try:
                            idx = int(raw_idx)
                        except (TypeError, ValueError):
                            idx = None
                        try:
                            weight = float(raw_weight)
                        except (TypeError, ValueError):
                            weight = None
                    elif isinstance(item, (list, tuple)) and len(item) >= 2:
                        try:
                            idx = int(item[0])
                        except (TypeError, ValueError):
                            idx = None
                        try:
                            weight = float(item[1])
                        except (TypeError, ValueError):
                            weight = None
                    if idx is None or weight is None or not math.isfinite(weight):
                        continue
                    usage[idx].append((weight, text))
    return usage


def build_glyph_semantics_from_usage(
    embed_fn: EmbedFn,
    usage_dict: Mapping[int, Sequence[tuple[float, str]]],
    *,
    glyph_count: int,
    target_dim: int | None = None,
    fallback_vectors: np.ndarray | None = None,
) -> np.ndarray:
    """Embed glyph semantics from usage text, weighted by glyph_topk contributions."""

    if glyph_count <= 0:
        return np.zeros((0, 0), dtype=np.float32)
    fallback = None
    if fallback_vectors is not None:
        fallback = np.asarray(fallback_vectors, dtype=np.float32)
        if fallback.ndim != 2:
            raise ValueError("fallback_vectors must be a 2-D array")
        if target_dim is None:
            target_dim = fallback.shape[1]

    glyph_vecs: list[np.ndarray] = []
    for g in range(glyph_count):
        items = usage_dict.get(g, [])
        if not items:
            if fallback is not None and g < fallback.shape[0]:
                glyph_vecs.append(fallback[g])
            else:
                glyph_vecs.append(np.zeros(target_dim or 0, dtype=np.float32))
            continue
        acc = None
        total_weight = 0.0
        for weight, text in items:
            if not text:
                continue
            vec = embed_fn(text)
            vec_arr = _prepare_vector(vec, target_dim=target_dim)
            if target_dim is None:
                target_dim = vec_arr.size
            if acc is None:
                acc = np.zeros(target_dim, dtype=np.float32)
            acc += float(weight) * vec_arr
            total_weight += float(weight)
        if acc is None or total_weight <= 0.0:
            if fallback is not None and g < fallback.shape[0]:
                glyph_vecs.append(fallback[g])
            else:
                glyph_vecs.append(np.zeros(target_dim or 0, dtype=np.float32))
        else:
            glyph_vecs.append(_normalize(acc / max(total_weight, 1e-9)))
    return np.stack(glyph_vecs, axis=0)


def blend_semantic_sources(
    basis_semantics: np.ndarray | None,
    usage_semantics: np.ndarray | None,
    *,
    usage_weight: float = 0.6,
) -> np.ndarray:
    """Blend Option A (basis) and Option B (usage) vectors."""

    if basis_semantics is None and usage_semantics is None:
        return np.zeros((0, 0), dtype=np.float32)
    if basis_semantics is None:
        return np.asarray(usage_semantics, dtype=np.float32)
    if usage_semantics is None:
        return np.asarray(basis_semantics, dtype=np.float32)
    basis = np.asarray(basis_semantics, dtype=np.float32)
    usage = np.asarray(usage_semantics, dtype=np.float32)
    if basis.shape != usage.shape:
        raise ValueError("basis_semantics and usage_semantics must have the same shape to blend")
    alpha = float(np.clip(usage_weight, 0.0, 1.0))
    mixed = ((1.0 - alpha) * basis) + (alpha * usage)
    norms = np.linalg.norm(mixed, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return mixed / norms


def save_glyph_semantics(
    glyph_vecs: np.ndarray,
    path: str | os.PathLike[str] = DEFAULT_GLYPH_SEMANTIC_PATH,
) -> Path:
    path = Path(path)
    _ensure_parent_dir(path)
    np.savez(path, glyphs=np.asarray(glyph_vecs, dtype=np.float32))
    return path


def load_glyph_semantics(
    path: str | os.PathLike[str] = DEFAULT_GLYPH_SEMANTIC_PATH,
) -> np.ndarray:
    data = np.load(Path(path), allow_pickle=False)
    return np.asarray(data["glyphs"], dtype=np.float32)


def _row_normalize(mat: np.ndarray) -> np.ndarray:
    arr = np.asarray(mat, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return arr / norms


def build_glyph_to_emoji_map(
    glyph_vecs: np.ndarray,
    emoji_vecs: np.ndarray,
    emoji_list: Sequence[str],
    *,
    topk: int = 3,
) -> dict[int, list[dict[str, float | str]]]:
    glyph_vecs = _row_normalize(glyph_vecs)
    emoji_vecs = _row_normalize(emoji_vecs)
    sims = glyph_vecs @ emoji_vecs.T
    mapping: dict[int, list[dict[str, float | str]]] = {}
    if sims.size == 0:
        return mapping
    for g in range(sims.shape[0]):
        row = sims[g]
        order = np.argsort(row)[::-1][:max(1, topk)]
        mapping[g] = [
            {"emoji": str(emoji_list[idx]), "score": float(row[idx])}
            for idx in order
        ]
    return mapping


def save_glyph_to_emoji_map(
    mapping: Mapping[int | str, Sequence[Mapping[str, float | str]]],
    path: str | os.PathLike[str] = DEFAULT_GLYPH_TO_EMOJI_PATH,
) -> Path:
    path = Path(path)
    _ensure_parent_dir(path)
    serializable: dict[str, list[dict[str, float | str]]] = {}
    for key, vals in mapping.items():
        serializable[str(key)] = [
            {"emoji": str(v.get("emoji")), "score": float(v.get("score", 0.0))}
            for v in vals
        ]
    with path.open("w", encoding="utf-8") as handle:
        json.dump(serializable, handle, ensure_ascii=False, indent=2)
    return path


def load_glyph_to_emoji_map(
    path: str | os.PathLike[str] = DEFAULT_GLYPH_TO_EMOJI_PATH,
) -> dict[str, list[dict[str, float | str]]]:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    normalized: dict[str, list[dict[str, float | str]]] = {}
    for key, items in raw.items():
        norm_items: list[dict[str, float | str]] = []
        for item in items:
            emoji = str(item.get("emoji", ""))
            if not emoji:
                continue
            score = float(item.get("score", 0.0))
            norm_items.append({"emoji": emoji, "score": score})
        normalized[str(key)] = norm_items
    return normalized


def glyph_topk_to_emojis(
    glyph_topk: Sequence[Sequence[float]],
    mapping: Mapping[int | str, Sequence[Mapping[str, float | str]]],
    *,
    per_glyph: int = 1,
) -> list[dict[str, float | str | int]]:
    """Convert glyph_topk entries to emoji payloads for the UI."""

    if not glyph_topk or not mapping:
        return []
    results: list[dict[str, float | str | int]] = []
    for entry in glyph_topk:
        if isinstance(entry, Mapping):
            glyph_idx = entry.get("glyph") or entry.get("index") or entry.get("id")
            weight = entry.get("weight") or entry.get("w") or entry.get("score")
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
            glyph_idx, weight = entry[0], entry[1]
        else:
            continue
        try:
            glyph_int = int(glyph_idx)
            weight_val = float(weight)
        except (TypeError, ValueError):
            continue
        candidates = mapping.get(str(glyph_int)) or mapping.get(glyph_int)
        if not candidates:
            continue
        for cand in list(candidates)[: max(1, per_glyph)]:
            emoji = cand.get("emoji")
            if not emoji:
                continue
            results.append(
                {
                    "glyph": glyph_int,
                    "emoji": str(emoji),
                    "weight": weight_val,
                    "score": float(cand.get("score", 0.0)),
                }
            )
    return results


__all__ = [
    "EMOJI_CANDIDATES",
    "DEFAULT_EMOJI_EMBED_PATH",
    "DEFAULT_GLYPH_SEMANTIC_PATH",
    "DEFAULT_GLYPH_TO_EMOJI_PATH",
    "build_emoji_embedding_table",
    "save_emoji_embeddings",
    "load_emoji_embeddings",
    "build_glyph_semantics_from_basis",
    "load_glyph_usage_from_logs",
    "build_glyph_semantics_from_usage",
    "blend_semantic_sources",
    "save_glyph_semantics",
    "load_glyph_semantics",
    "build_glyph_to_emoji_map",
    "save_glyph_to_emoji_map",
    "load_glyph_to_emoji_map",
    "glyph_topk_to_emojis",
]
