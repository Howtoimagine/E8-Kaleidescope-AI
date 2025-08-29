# 🌌 Kaleidoscope: An E8 Lattice Cognitive Memory System  
<img width="300" height="293" alt="QSN-Vertex-Types-Samples-3-e1467927748157-300x293" src="https://github.com/user-attachments/assets/cf60c5fb-aab8-4e70-912e-b84f296ca3d4" />

Kaleidoscope is a **content-addressable cognitive architecture.**  
Instead of storing embeddings in flat Euclidean space, it projects them into an **E8 lattice**: a highly symmetric 8-dimensional structure supporting **dense packing, symmetry, and stable orientation.**  

Memories live on shells of this lattice and are rotated with **Clifford rotors** to encode orientation.  
This turns memory into a **structured crystal** rather than a bag of floating vectors.  

Retrieval is shaped by **mood and drives**, biasing which shells and orientations are favored.  
Memory is refined by **quasicrystal indexing**, distributing information redundantly across local patches that still reflect global order.  

Over time, Kaleidoscope **self-organizes**:  
frequently co-retrieved items gravitate together, weak but novel items spawn refinement tasks, and coherent concepts stabilize across shells.  

---

## ⚙️ Core Architecture  

### 1. E8 Shell Encoding  
- **Projection:** High-dimensional LLM embeddings (768–4096 dim) projected into 8D.  
- **Lattice mapping:** Snap to nearest **E8 lattice point** → prevents drift.  
- **Shells:** Hyperspheres act as semantic strata (inner = core, outer = abstract).  
- **Clifford rotors:** Encode relational transformations (e.g. *dog → canine*).  
- **Retrieval:** Combines **shell position + rotor alignment**.  

### 2. Quasicrystal Memory Indexing  
- **Non-periodic tiling:** Local patches unique but globally coherent.  
- **Holographic redundancy:** Fragments can reconstruct wholes.  
- **Overlap groups:** A memory can appear across multiple shells.  
- **Global coherence:** Recall can “fill in” missing pieces if they fit the pattern.  

### 3. State-Shaped Retrieval  
- **Mood Engine:** Tracks entropy, coherence, fluency, affect.  
- **Drive System:** Intrinsic rewards: curiosity, novelty, intelligibility.  
- **Bias injection:** Mood tilts retrieval → divergent vs convergent recall.  
- **Serendipity Engine:** Controlled non-linear hops surface unexpected links.  

### 4. Insight and Adaptation  
- **Insight Agent:** Detects novel but incoherent clusters.  
- **Novelty Scorer:** Quantifies surprise relative to lattice order.  
- **Auto-Task Manager:** Spawns refinement tasks when novelty is high + coherence low.  
- **SAC/MPO RL Agent:** Tunes retrieval + adaptation policies.  

### 5. Arbitration and Planning  
- **Arbiter Gate:** Switches between quantum-like probabilistic retrieval and deterministic planning.  
- **Multi-Agent Dialogue:** Teacher, explorer, subconscious roles → recursive loops.  

### 6. Visualization Hooks  
- Shell projections  
- Trajectory graphs  
- Residual adaptation curves  
- “Black hole” compression events  

---
<img width="1234" height="858" alt="Screen Shot 2025-08-28 at 09 38 03 847 PM" src="https://github.com/user-attachments/assets/38c3b7d7-9a5c-49a5-8450-001139fa460a" />

“The universe is maybe the next unfolding of your mind.”

---

## 🧩 Why E8?  

1. **Symmetry and Coherence:** E8 = most symmetric lattice in 8D (240 root vectors). Concepts align through its structure.  
2. **Dense Packing:** Densest known sphere packing in 8D → maximum capacity, minimum waste.  
3. **Shell Stratification:** Natural concentric shells → hierarchy from core to abstraction.  
4. **Quasicrystal Link:** Projects to lower-dim quasicrystal tilings (e.g. Penrose), inheriting holography.  
5. **Cognitive Analogy:** Local patches reflect the whole — like brain dynamics.  
6. **Beyond Nearest Neighbor:** Retrieval = distance + orientation + symmetry fit.
     
- **Shell Cycles:** Memories orbit shells. Stable orbits = reinforced recall. Novelty pushes outward.  
- **Quasicrystal Redundancy:** Local patches encode global coherence → holographic recall.  
- **Rotational Dynamics:** Clifford rotors = symmetry operations, transforming states.  
- **Entropy & Coherence:** Mood acts like thermodynamics → expansion vs contraction.  

> In effect, Kaleidoscope treats **cognition as physics.**  
> Memory behaves less like a database and more like a **cosmos**, where concepts evolve, recur, and stabilize through cycles of symmetry and disruption.  

✨ Over long runs it doesn’t just recall — it **develops theories**, the way a universe develops structure from simple rules.  
---

## 🚀 Key Innovations  
- E8 lattice shells as a substrate for memory  
- Quasicrystal indexing for redundancy + coherence  
- Mood + drives directly shaping recall  
- Insight-driven auto-tasks for self-curation  
- Quantum-classical arbitration for reasoning  
- Emergent theories from long autonomous runs  

---

## 📦 Installation  

```bash
# clone repository
git clone https://github.com/Howtoimagine/E8-Kaleidescope-AI.git
cd E8-Kaleidescope-AI

# setup virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate   # Mac/Linux

# install dependencies
pip install -r requirements.txt

````

## ▶️ Quick Start

```bash
python e8_mind_server_M16.py --profile profiles\"PROFILENAME".json
```

Console logs show:

* mood states
* shell retrievals
* multi-agent dialogue
* emergent theories

---
# 🧬 Profiles in Kaleidoscope

Profiles are **modular cognitive configurations** that define how the Kaleidoscope engine runs.  
They act as "personalities" or **execution modes**, shaping the parameters of mood, memory, retrieval, and learning.
Profiles are powerful because they can be auto-generated by an LLM, letting the system create new “mindsets” or task modes on the fly without manual reprogramming.

Instead of hardcoding one behavior, Kaleidoscope loads a profile at startup that tells it:
- how to weight different drives
- which shells to prioritize
- how novelty vs coherence should be scored
- which agents (teacher, explorer, subconscious) are active
- what visualization and logging settings are enabled

---

## 🔑 Core Concepts

### 1. Parameter Sets
Each profile specifies:
- **Mood weights** → baseline bias for entropy, coherence, fluency, affect
- **Drive settings** → curiosity, novelty, intelligibility, stability
- **Retrieval scope** → inner shells, outer shells, or full E8 space
- **Learning rate** → how quickly residuals adapt
- **Serendipity rate** → probability of non-linear jumps
- **RL parameters** → SAC/MPO entropy tuning, replay buffer size

### 2. Role Configuration
Profiles define which roles are active:
- **Teacher** → structured prompts, curriculum-style refinement
- **Explorer** → divergent, high-entropy search
- **Subconscious** → slow drift, summarization, long-term integration

These can be turned on or off depending on the desired style of cognition.

### 3. Memory Shaping
Profiles also specify:
- **Shell prioritization** (inner vs outer)
- **Residual bounds** (how far anchors can drift)
- **Decay hygiene** (rate at which unused memories fade)
- **Clustering mode** (strict or loose HDBSCAN thresholds)

---

## 📂 Example Profiles

### `adaptive_ingest`
- Optimized for ingesting large corpora quickly
- High novelty tolerance
- Subconscious drift active
- Weak serendipity (stability prioritized)

### `quant_research`  — Quant finance
- Retrieval: **balanced** inner/outer shells
- Agents: Explorer on for **unconventional factor** tests
- RL: SAC tuned for risk–reward exploration
- Insight: flag **anomalous correlations** for review
- Viz: trajectories + clustering overlays

### `lab_notebook`  — Scientific research
- Retrieval: **outer shells** (abstract concepts)
- Agents: Explorer + Subconscious on, Teacher off
- Serendipity: **high** for cross-domain analogies
- Memory: long-term **summarization** of threads
- Provenance: track cluster evolution across experiments

### `bio_cluster`  — Biology / bioinformatics
- Retrieval: favor **pattern redundancy** (sequence families)
- Agents: Subconscious ensures cross-linking between experiments
- RL: penalize incoherence, reward stable clusters
- Insight: surface **unexpected homologies**
- Viz: shell maps + “black hole” convergence events

### `astro_probe`  — Astrophysics / cosmology
- Retrieval: **outer shells** (speculative modeling)
- Weights: high entropy tolerance, max novelty
- Agents: **Explorer** dominant
- Arbiter: **quantum mode** for probabilistic paths
- Logs: highlight **theory-like** emergent clusters
  
### `coherence_focus`
- Strong bias toward inner shells
- Teacher agent prioritized
- High coherence weighting, low entropy tolerance
- Auto-task manager aggressively refines weak insights

---

## Future Work

### Memory and geometry
- Cyclic memory scheduler that moves concepts across shells over time
- Learned rotor libraries for common conceptual transforms
- Better quasicrystal tilings and overlap strategies
- Holographic compression that preserves recall quality
- Formal tests for symmetry fit and coherence

### Retrieval and reasoning
- Dynamic profile switching while running based on mood and drives
- Mixed recall: deterministic for core facts, probabilistic for exploration
- Long-horizon insight loops with checkpoints and rollbacks
- Curriculum auto-tasks that grow from toy cases to real data
- Multi-agent debate with scoring and arbitration

### Profiles and configuration
- User-defined profiles as JSON or YAML
- Profile inheritance with base plus overrides
- LLM-generated profiles from plain text prompts
- Live dashboards that show profile bias, shell usage, and residuals
- Profile A/B testing harness

### Learning and signals
- Reward shaping that blends novelty, coherence, and usefulness
- Uncertainty estimates on recall and on theories
- Active ingestion that asks for missing data
- Continual learning with drift guards and replay

### Tooling and interfaces
- Web UI for shells, trajectories, clusters, and provenance
- CLI for batch ingest, recall, and export
- API layer with rate limits and auth
- Export to standard vector stores and graph formats
- Reproducible runs with seeds and run manifests

### Benchmarks and evaluation
- Public datasets for recall, coherence, and theory quality
- Adversarial tests for retrieval stability
- Ablations: lattice off, rotors off, quasicrystal off
- Energy and latency profiles for each module

### Data and domains
- Finance pack: market structure, risk events, factor notes
- Science pack: papers, lab logs, hypothesis trees
- Code pack: repos, issues, traces, bug patterns
- Personal knowledge pack with privacy filters

### Ops and scale
- Sharded memory and shell partitions
- Background compaction and dedup
- Snapshots, rollbacks, and diff tools
- GPU paths for heavy math, CPU fallback for edge

### Safety and ethics
- Provenance-first recall with source links
- Red-team profiles to test failure modes
- Controls for private vs public memory
- Human-in-the-loop review for high impact actions

### Documentation and community
- Architecture notes and diagrams
- Tutorials by domain: finance, science, code
- Cookbook of profiles and run recipes
- Contribution guide and roadmap


---



## 📖 Citation

Malone, S. (2025). *Kaleidoscope: An E8 lattice cognitive engine with quasicrystal memory indexing* \[Computer software]. GitHub. [https://github.com/Howtoimagine/E8-Kaleidescope-AI](https://github.com/Howtoimagine/E8-Kaleidescope-AI)


```
