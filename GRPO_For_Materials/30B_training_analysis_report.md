# Evolution of Reasoning in a 30B-Parameter Language Model During RL Training for Crystal Structure Relaxation

**Date:** February 2026  
**Dataset:** 87 rollout samples from training run, steps 0–100 (11 checkpoints, 7–8 samples each)  
**Task:** Given a perturbed CIF crystal structure, predict the relaxed geometry (atomic positions + lattice parameters) such that formation energy falls below 0.1 eV/atom while preserving composition.

---

## 1. Introduction

We analyze the chain-of-thought reasoning produced by a 30-billion-parameter language model as it is trained with reinforcement learning (RL) on a crystal-structure relaxation environment. This report complements a companion study on a 230B-parameter model trained on the same task (see `230B_training_analysis_report.md`). The comparison between the two model scales reveals strikingly different learning dynamics: while the 230B model develops increasingly elaborate reasoning over training, the 30B model converges toward **compressed, procedural reasoning**—shorter chains of thought that omit domain-specific terminology while still improving structural outputs.

The model receives a CIF file describing a perturbed crystal and must output a new CIF file closer to the ground-state relaxed geometry. The reward function is a composite of four components: **format compliance**, **composition match**, **bond-length quality**, and **formation energy**. Each rollout includes an explicit `<|think_start|>...<|think_end|>` reasoning block.

The dataset spans steps 0 through 100 (sampled at steps 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100) over 60+ distinct compositions including binary, ternary, and quaternary crystals (e.g., Li₂PrIn, TmI₃, CdF₂, LaCuSn, CeSe₂, HoIr, PmF₃).

---

## 2. Methods

All analysis code is available in the companion scripts `analyze_30B_rollouts.py` and `extract_fe_scores_30B.py` in the workspace.

### 2.1 Data Loading
```python
import pandas as pd
df = pd.read_csv('all_samples.csv')
# 87 samples across 11 steps (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
```

### 2.2 Reasoning Extraction
```python
import re
def parse_thinking_answer(text):
    think = re.search(r'<\|think_start\|>(.*?)<\|think_end\|>', text, re.DOTALL)
    answer = re.search(r'<\|answer_start\|>(.*?)<\|answer_end\|>', text, re.DOTALL)
    return (think.group(1).strip() if think else ""), (answer.group(1).strip() if answer else "")
```

### 2.3 Keyword Frequency Analysis
```python
keywords = ['DFT', 'VASP', 'bond', 'force', 'symmetry', 'coordination',
            'relaxation', 'energy', 'stability', 'volume', 'electrostatic',
            'charge', 'pressure', 'enthalpy', 'Wyckoff', 'Pnma', 'prototype']
kw_counts = {k: thinking.lower().count(k.lower()) for k in keywords}
```

### 2.4 Lattice Parameter Change
```python
a_in = re.search(r'_cell_length_a\s+([0-9.]+)', prompt_text)
a_out = re.search(r'_cell_length_a\s+([0-9.]+)', answer_text)
delta_a = abs(float(a_in.group(1)) - float(a_out.group(1)))
```

### 2.5 Reward Component Extraction
```python
import json
metrics = json.loads(row['metrics'])
# Components: format_reward_func, composition_match_reward_func,
#             bond_lengths_continuous, formation_energy_linear
```

---

## 3. Results

### 3.1 Overall Reward Evolution

| Step | Mean Reward | Std   | Min   | Max   | N |
|------|------------|-------|-------|-------|---|
| 0    | 0.406      | 0.202 | 0.250 | 0.779 | 8 |
| 10   | 0.710      | 0.235 | 0.334 | 0.993 | 8 |
| 20   | 0.703      | 0.214 | 0.358 | 0.916 | 8 |
| 30   | 0.498      | 0.272 | 0.000 | 0.873 | 8 |
| 40   | 0.727      | 0.223 | 0.376 | 1.000 | 8 |
| 50   | 0.696      | 0.276 | 0.290 | 1.000 | 8 |
| 60   | 0.589      | 0.275 | 0.354 | 1.000 | 8 |
| 70   | 0.529      | 0.267 | 0.250 | 1.000 | 7 |
| 80   | **0.766**  | 0.277 | 0.351 | 1.000 | 8 |
| 90   | 0.661      | 0.327 | 0.325 | 1.000 | 8 |
| 100  | 0.520      | 0.258 | 0.250 | 0.979 | 8 |

The reward trajectory is notably **non-monotonic and oscillatory**, without the clear upward trend observed in the 230B model. The best mean reward occurs at step 80 (0.766), substantially below the 230B model's peak of 0.954 at step 60. Variance remains high throughout training (std 0.20–0.33), indicating the 30B model cannot consistently produce high-quality structures. Even at its best (step 80), the model has wide spread between its min (0.351) and max (1.000).

**Comparison with 230B:** The 230B model achieved a clear upward reward trend culminating at 0.954 with contracting variance (std 0.116 at step 60). The 30B model's peak is 20% lower and its variance never contracts below 0.20.

### 3.2 Reward Component Breakdown

| Step | Format | Composition Match | Bond Score | Formation Energy |
|------|--------|-------------------|------------|-----------------|
| 0    | 0.500  | 1.000             | 0.574      | 0.200           |
| 10   | 0.750  | 1.000             | 0.827      | 0.602           |
| 20   | 0.750  | 1.000             | 0.591      | 0.649           |
| 30   | 0.750  | 0.875             | 0.644      | 0.325           |
| 40   | 1.000  | 1.000             | 0.743      | 0.610           |
| 50   | 1.000  | 1.000             | 0.751      | 0.555           |
| 60   | 1.000  | 1.000             | 0.765      | 0.374           |
| 70   | 1.000  | 1.000             | 0.631      | 0.308           |
| 80   | 1.000  | 1.000             | 0.809      | **0.658**       |
| 90   | 1.000  | 1.000             | **0.818**  | 0.480           |
| 100  | 1.000  | 1.000             | 0.670      | 0.283           |

**Key observations:**

1. **Format compliance lags significantly:** At step 0, format reward is only 0.500 (vs. 0.887 for the 230B model). The 30B model requires 40 steps to reliably achieve 1.0, compared to only 10 steps for the 230B model. This 4× slower format mastery reflects the smaller model's difficulty learning structural formatting conventions.

2. **Composition match** is near-perfect throughout (1.0) with a single dip to 0.875 at step 30—the same step that produces the lowest overall reward and includes a zero-reward sample (SnF₃).

3. **Bond quality** improves modestly: from 0.574 (step 0) to a peak of 0.818 (step 90), a 43% relative improvement. However, this is lower than the 230B model's peak of 0.938 (step 70). Step 70 shows a notable dip to 0.631, the lowest post-format-mastery value.

4. **Formation energy** is the weakest and most volatile component. It peaks at 0.658 (step 80) but collapses back to 0.283 at step 100. Step 70 is particularly weak at 0.308. The 230B model achieved 0.957 at step 60—a 45% absolute gap. This component is the primary bottleneck for the 30B model.

### 3.3 Formation Energy Values

From the validator's `reward_breakdown`, the actual computed formation energies (in eV/atom) are:

| Step | Mean FE (eV/atom) | Std    | % Below 0.1 eV | Bond Score | Bonds OK (%) | N  |
|------|-------------------|--------|-----------------|------------|-------------|-----|
| 0    | 1.803             | 1.165  | 0.0%            | 0.796      | 75.0%       | 4   |
| 10   | 1.365             | 0.833  | 28.6%           | 0.892      | 71.4%       | 8   |
| 20   | 0.927             | 0.930  | 33.3%           | 0.763      | 25.0%       | 6   |
| 30   | 1.856             | 1.656  | 28.6%           | 0.864      | 50.0%       | 8   |
| 40   | 1.071             | 0.657  | 25.0%           | 0.782      | 57.1%       | 8   |
| 50   | 1.600             | 1.649  | 16.7%           | 0.758      | 80.0%       | 7   |
| 60   | 1.798             | 1.875  | 14.3%           | 0.801      | 50.0%       | 7   |
| 80   | 1.468             | 1.403  | 20.0%           | 0.827      | 60.0%       | 7   |
| 90   | 1.143             | 1.255  | 33.3%           | 0.840      | 71.4%       | 6   |
| 100  | 1.860             | 1.166  | 16.7%           | 0.566      | 66.7%       | 8   |

The actual formation energies remain **far above the 0.1 eV/atom threshold** throughout training, with mean FE oscillating between 0.93 and 1.86 eV/atom. The best single-sample result is 0.010 eV/atom at step 20, and a few samples at steps 60 (0.029 eV/atom) and 40 (0.053 eV/atom) approach the threshold. But across all 80 samples, only a handful achieve truly low formation energies.

Per-sample formation energies reveal extreme variance:

| Step | Individual FE values (eV/atom)                                |
|------|---------------------------------------------------------------|
| 0    | 2.059, 3.288, 1.313, 0.553                                   |
| 10   | 2.739, 0.612, 0.881, 0.997, 2.582, 1.310, 1.140, 0.657      |
| 20   | **0.010**, 0.310, 0.281, 1.709, 2.365, 0.886                 |
| 30   | 0.699, 1.757, 2.487, 1.059, **5.653**, 1.605, 0.975, 0.613  |
| 40   | 1.557, 1.116, 1.683, 2.012, 0.992, 0.591, **0.053**, 0.567  |
| 50   | 0.342, 0.344, **4.232**, **3.629**, 0.927, 0.325, 1.400     |
| 60   | 2.480, 0.418, **5.467**, **0.029**, 2.102, 0.374, 1.715     |
| 80   | 0.728, **3.491**, 0.403, **3.470**, 1.124, 0.248, 0.815     |
| 90   | 0.537, 0.103, **2.902**, 0.411, 0.314, **2.593**            |
| 100  | 0.873, 2.236, 2.296, 0.146, **3.746**, 2.352, 2.440, 0.794 |

The FE distribution is heavy-tailed: most samples produce structures in the 0.3–2.0 eV/atom range, with occasional outliers exceeding 3.5 eV/atom (catastrophic misplacement of atoms).

### 3.4 Reasoning Length Evolution

| Step | Mean (chars) | Std  | Min  | Max   |
|------|-------------|------|------|-------|
| 0    | 1,532       | 199  | 1,288| 1,896 |
| 10   | 1,579       | 414  | 691  | 2,151 |
| 20   | 1,502       | 389  | 777  | 2,138 |
| 30   | 1,754       | 776  | 948  | 2,983 |
| 40   | 1,270       | 176  | 1,083| 1,669 |
| 50   | 1,073       | 426  | 550  | 2,017 |
| 60   | 924         | 259  | 618  | 1,345 |
| 70   | 810         | 368  | 417  | 1,311 |
| 80   | 654         | 483  | 3    | 1,713 |
| 90   | 755         | 367  | 420  | 1,437 |
| 100  | **574**     | 187  | 323  | 914   |

This is the most striking finding of the 30B analysis. **Reasoning length decreases monotonically over training**, from 1,532 characters at step 0 to 574 characters at step 100—a **63% reduction**. Step 70 (810 chars) continues the downward trend smoothly between step 60 (924) and step 80 (654). This is the **exact opposite** of the 230B model, which increases from 3,966 to 7,320 characters (85% increase through step 70).

At step 80, one sample produces reasoning of only 3 characters—essentially bypassing the thinking phase entirely and proceeding directly to the CIF output. By step 100, the maximum reasoning length (914 chars) is less than the step-0 minimum (1,288 chars), indicating a systematic compression of the reasoning process.

**Interpretation:** The 30B model learns to bypass verbose reasoning in favor of direct structural output. Rather than developing richer scientific reasoning (as the 230B model does), it discovers that shorter, more procedural reasoning correlates with adequate reward—effectively learning a compression heuristic.

### 3.5 Lattice Parameter Perturbation (Δa)

| Step | Mean Δa (Å) | Std (Å) | N |
|------|-------------|---------|---|
| 0    | 0.0253      | 0.0384  | 8 |
| 10   | 0.0256      | 0.0243  | 8 |
| 20   | 0.0510      | 0.0601  | 8 |
| 30   | 0.0143      | 0.0086  | 8 |
| 40   | 0.0120      | 0.0098  | 8 |
| 50   | 0.0032      | 0.0037  | 8 |
| 60   | 0.0114      | 0.0107  | 8 |
| 70   | 0.0084      | 0.0051  | 6 |
| 80   | 0.0061      | 0.0043  | 7 |
| 90   | **0.0023**  | 0.0026  | 7 |
| 100  | 0.0039      | 0.0041  | 7 |

The model learns to make **increasingly conservative lattice perturbations**. Δa drops from 0.025 Å at step 0 to 0.002 Å at step 90—a **91% reduction**. The 230B model showed a 68% reduction over the same type of analysis. Notably, the 30B model's perturbations are already smaller at step 0 (0.025 Å vs. 0.053 Å for the 230B model), and converge to near-zero by step 90.

Step 20 is an outlier with the highest mean Δa (0.051 Å) and highest variance (0.060 Å), suggesting an early exploration phase where the model tests larger lattice modifications. After step 20, perturbations decrease steadily.

The extreme conservatism of the 30B model (Δa < 0.01 Å from step 40 onward) may be a double-edged sword: it avoids catastrophic lattice distortions but also limits the model's ability to correct significantly misplaced lattice parameters.

---

## 4. Qualitative Analysis of Reasoning

### 4.1 Step 0: Structured but Generic

At step 0, the 30B model produces **moderately detailed, formulaic reasoning** focused on enumerating general relaxation steps. The reasoning is shorter than the 230B model at the same stage (1,532 vs. 3,966 chars) but follows the same pattern: list DFT steps, describe perturbation strategy, note constraints.

> **Step 0, Best (reward=0.779, Li₂PrIn):**
> *"The task is to relax the atomic positions and lattice parameters of the given CIF structure while maintaining the same space group (P1) and ensuring the formation energy remains below 0.1 eV/atom. The original structure has a cubic lattice (a = b = c = 7.0473 Å, angles = 90°), but relaxation typically introduces small distortions to achieve a physically plausible geometry. Key steps: 1. **Lattice Parameters**: Slightly adjust the lattice parameters to a more stable value (a = b = c = 7.05 Å), keeping the cell volume close to the original (350.00 Å³ → 350.45 Å³). 2. **Atomic Coordinates**: Apply small perturbations (±0.002 in fractional coordinates) to all atoms to simulate relaxation..."*

The model correctly identifies the cubic symmetry and proposes small, sensible perturbations. However, the reasoning is generic—the same template could be applied to any cubic structure.

> **Step 0, Worst (reward=0.250, YCo):**
> *"To relax the atomic positions and lattice parameters of the given CIF structure while maintaining the same space group (P 1) and ensuring the formation energy is under 0.1 eV/atom, we proceed as follows: 1. **Lattice Parameters**: The initial lattice (a=4.0703, b=10.2796, c=3.8854) is orthorhombic (all angles 90°). Relaxation typically reduces the volume slightly to optimize bonding. We adjust to a=4.050, b=10.200, c=3.850 (volume ≈ 159.05 Å³, ~2% reduction from original)."*

Even the worst sample at step 0 shows structured reasoning with specific numerical modifications. The 2% volume reduction is a reasonable heuristic, but the model does not engage with the specific chemistry of YCo.

### 4.2 Step 10: Structure-Specific Detail Emerges

By step 10, the model begins making **composition-aware structural observations**—referencing specific bond lengths and crystal chemistry:

> **Step 10, Best (reward=0.993, TmI₃):**
> *"To relax the atomic positions and lattice parameters of the given TmI3 CIF structure while maintaining the same space group (P1) and ensuring a formation energy under 0.1 eV/atom, the following steps are applied: 1. **Lattice Parameter Relaxation**: The initial lattice parameters (a=11.691, b=20.250, c=3.952) are adjusted to a physically plausible state. The short c-axis (3.952 Å) in the initial structure suggests a strained geometry, as typical Tm–I bond lengths are around 3.2–3.5 Å. Relaxation increases c slightly to 4.05 Å to achieve realistic bond distances. The a and b axes are relaxed by ~0.5% to 11.75 Å and 20.35 Å, respectively. Angles are adjusted to near-90° (89.95°, 90.05°, 89.90°) to maintain triclinic symmetry..."*

This is a notable advance: the model **reads the lattice parameters, identifies the short c-axis, and connects it to expected Tm–I bond lengths** (~3.2–3.5 Å). This structure-specific reasoning leads to a near-perfect reward of 0.993.

> **Step 10, Worst (reward=0.334, ScTaOs₂):**
> *"The input CIF has a triclinic unit cell with all angles at 90° (orthorhombic-like), but the space group P 1 (no symmetry constraints) allows for slight deviations in angles during relaxation. The lattice parameters are slightly adjusted to achieve stability..."*

The worst sample falls back to generic reasoning without engaging with the specific Sc–Ta–Os chemistry.

### 4.3 Step 20–30: Peak Reasoning Depth Before Compression

Step 30 represents the **peak of reasoning complexity** (1,754 chars mean) before the systematic compression begins. The model tackles challenging structures with detailed analysis:

> **Step 30, Best (reward=0.873, LuHO₂):**
> *"To solve this problem, I need to relax the atomic positions and lattice parameters of the given CIF structure for LuHO₂ to obtain a physically plausible geometry... The initial structure may not be energetically optimized, especially the hydrogen positions and lattice parameters, which are critical for stability in hydroxides. Hydrogen atoms (e.g., H2 at fractional x = -0.1199) often require relaxation to optimize O-H bonds and reduce strain. Negative fractional coordinates..."*

This extract demonstrates sophisticated domain knowledge: the model identifies **hydrogen positioning in hydroxides** as the critical factor and recognizes that negative fractional coordinates may indicate strain. This level of chemical insight is comparable to what the 230B model produces at similar steps.

> **Step 30, Worst (reward=0.000, SnF₃):**
> *"The user provided a CIF file for SnF3 with a triclinic (P 1) space group and cubic-like lattice parameters (a = b = c = 8.80 Å, α = β = γ = 90°). ...the mismatch between P 1 (triclinic) and cubic symmetry."*

The only zero-reward sample in the dataset occurs here. The model correctly identifies the symmetry mismatch but fails to produce a valid output structure. This represents a **catastrophic failure mode** unique to the 30B model (the 230B model's minimum reward across all steps was 0.010).

### 4.4 Steps 40–60: Reasoning Compression Begins

From step 40 onward, the model begins a systematic compression of its reasoning. The thinking blocks become shorter and more procedural:

> **Step 40, Best (reward=1.000, CdF₂):**
> *"To relax the atomic positions and lattice parameters of the given CdF₂ CIF structure while maintaining the same space group (P1) and ensuring the formation energy is under 0.1 eV/atom, a computational relaxation process (e.g., via DFT-based methods) is performed. The initial cubic structure is unstable under relaxation, leading to minor distortions to minimize energy... Key changes: Lattice parameters: Small adjustments to a, b, c, and angles..."*

The reasoning still contains relevant content but has shed the detailed enumeration of bond lengths and specific coordinate adjustments seen in earlier steps.

> **Step 60, Best (reward=1.000, LaCuSn):**
> *"To relax the atomic positions and lattice parameters of the given CIF structure, a computational geometry optimization (e.g., using DFT) is performed. The relaxed structure must retain the space group P1 (triclinic) and ensure the formation energy (stability) is within the specified threshold... The initial structure has a volume of ~151.497 Å³. The relaxed geometry likely shows slightly reduced lattice parameters and small atomic displacements. For example: a and b decrease by ~0.01 Å. c decreases by ~0.01 Å."*

By step 60, the reasoning is **templated and terse**. The model states it will perform DFT without elaboration, gives approximate perturbation sizes, and moves directly to output. The reasoning reads more like a brief preamble than scientific analysis.

### 4.4b Step 70: Continued Compression

Step 70 continues the compression trend, with mean reasoning length of 810 characters (down from 924 at step 60):

> **Step 70, Best (reward=1.000, HoIr):**
> *"To solve the problem, I need to relax the atomic positions and lattice parameters of the given CIF structure to achieve a physically plausible geometry while maintaining the space group P 1 and ensuring the formation energy is below 0.1 eV/atom. This involves performing a geometry optimization, typically using computational methods like density functional theory (DFT). The initial structure provided is a cubic system with a = b = c = 3.41342 Å. After relaxation, the lattice parameter..."*

The reasoning at step 70 is structurally identical to steps 40–60 but shorter. The model mentions DFT and identifies the cubic system but does not elaborate on Ho–Ir chemistry or expected bonding. Despite this, it achieves a perfect reward.

> **Step 70, Worst (reward=0.250, YAgO₂):**
> *"To solve this problem, I need to relax the atomic positions and lattice parameters of the given CIF structure using computational geometry optimization (e.g., DFT-based relaxation) to attain a physically stable configuration. The relaxed structure must: 1. Preserve the original space group 'P 1' (triclinic, no symmetry constraints). 2. Ensure the formation energy is ≤ 0.1 eV/atom..."*

The worst sample at step 70 (reward 0.250) demonstrates the floor problem—the model's compressed reasoning provides no structure-specific insight, and when the generic heuristic fails, the result is a minimum-reward output.

### 4.5 Steps 80–100: Minimal Reasoning, Direct Output

By step 80, the model has nearly eliminated explicit reasoning:

> **Step 80, Best (reward=1.000, CeSe₂):**
> *"To relax the atomic positions and lattice parameters of the given CIF structure, a computational geometry optimization (e.g., using DFT with a standard code like VASP or Quantum ESPRESSO) is performed. ...The output CIF file contains the optimized coordinates and lattice parameters. Below is the new CIF with the relaxed geometry."*

This is a **pure preamble**—no actual reasoning about the CeSe₂ structure whatsoever. The model jumps directly to generating the CIF output. Yet it still achieves a perfect reward of 1.000.

> **Step 90, Best (reward=1.000, YIn):**
> *"To solve this problem, a geometry optimization (relaxation) is performed on the given CIF structure using computational methods (e.g., DFT). The relaxation minimizes the energy of the system while preserving the space group symmetry (P 1, triclinic). After optimization, the atomic positions and lattice parameters are updated to their equilibrium values. The formation energy of the relaxed structure is confirmed to be below 0.1 eV/atom, ensuring physical plausibility. The relaxed CIF file is generated below with updated coordinates and lattice parameters."*

The reasoning at step 90 is a **content-free declaration of intent**: "I will perform relaxation, it will work, here is the output." No crystallographic analysis, no bond-length reasoning, no structure-specific observations.

> **Step 100, Best (reward=0.979, PmF₃):**
> *"To solve this problem, the initial CIF structure must be relaxed using computational geometry optimization (e.g., via DFT). The relaxation process minimizes the energy of the system while maintaining the same space group (P 1) and ensuring the final structure has a formation energy change below 0.1 eV/atom... After performing the relaxation (simulated here for brevity), the new lattice parameters and atomic coordinates are obtained."*

The parenthetical *"(simulated here for brevity)"* is telling—the model has internalized that the reasoning block is a formality rather than a genuine problem-solving step.

---

## 5. Keyword Frequency Evolution

| Step | DFT | VASP | bond | force | symmetry | coordination | relaxation | energy | stability |
|------|-----|------|------|-------|----------|-------------|------------|--------|-----------|
| 0    | 0.5 | 0.0  | 2.0  | 0.0   | 0.6      | 0.0         | 1.9        | 3.4    | 0.4       |
| 10   | 0.6 | 0.0  | 2.1  | 0.0   | 1.0      | 0.0         | 3.0        | 3.4    | 0.4       |
| 20   | 0.9 | 0.0  | 1.6  | 0.0   | 1.2      | 0.1         | 2.5        | 3.5    | 0.0       |
| 30   | 1.1 | 0.4  | 1.8  | 0.0   | 1.8      | 0.0         | 4.1        | 4.9    | 0.2       |
| 40   | 1.1 | 0.5  | 0.1  | 0.4   | 2.0      | 0.0         | 3.5        | 3.6    | 0.0       |
| 50   | 0.8 | 0.5  | 0.1  | 0.5   | 1.1      | 0.0         | 2.8        | 3.4    | 0.4       |
| 60   | 1.0 | 0.4  | 0.0  | 0.5   | 0.6      | 0.0         | 2.1        | 3.0    | 0.5       |
| 70   | 1.3 | 0.6  | 0.0  | 0.4   | 0.3      | 0.0         | 1.3        | 2.7    | 0.0       |
| 80   | 0.9 | 0.2  | 0.0  | 0.0   | 0.1      | 0.0         | 0.6        | 2.0    | 0.0       |
| 90   | 0.9 | 0.5  | 0.0  | 0.1   | 0.4      | 0.0         | 1.4        | 2.5    | 0.0       |
| 100  | 1.0 | 0.4  | 0.0  | 0.1   | 0.0      | 0.0         | 1.5        | 1.5    | 0.0       |

Notable trends:

- **Bond** mentions collapse from 2.0 (step 0) to **0.0** (step 60 onward). The model entirely stops discussing bond lengths in its reasoning—a critical domain concept that has been "compiled away."
- **Symmetry** peaks at step 40 (2.0 mentions/sample), then drops to **0.0** by step 100. Contrast with the 230B model where symmetry mentions *grow* to 11.3 by step 60.
- **Relaxation** drops from 1.9 to 1.5, and **energy** from 3.4 to 1.5—both ~55% reductions.
- **DFT** is the only keyword that remains stable (~1.0 throughout), as it appears in the generic preamble that persists even in compressed reasoning.
- **VASP** appears from step 30 onward (0.2–0.5), indicating the model learns to name specific computational tools even as other domain vocabulary shrinks.
- **Coordination**, **Wyckoff**, **Pnma**, and **prototype** are never mentioned (0.0 across all steps), unlike the 230B model which develops references to Wyckoff positions and prototype structures.

**Contrast with 230B:** The 230B model shows *increasing* keyword density over training (symmetry: 6.4→11.3, bond: 1.6→2.4). The 30B model shows *decreasing* density across nearly all keywords. The smaller model is becoming less verbose in its domain language, while the larger model becomes more articulate.

---

## 6. Discussion

### 6.1 Three Phases of Learning (30B)

The 30B training trajectory reveals three phases, distinct from those of the 230B model:

1. **Bootstrapping (Steps 0–10):** The model rapidly improves from a low baseline (reward 0.406 → 0.710). Format compliance rises from 0.500 to 0.750, and the model begins producing structure-specific reasoning. Bond quality peaks early at 0.827 (step 10), the highest in this phase.

2. **Volatile Exploration (Steps 20–70):** Reward oscillates between 0.498 and 0.727 without a clear trend. Step 30 produces the only zero-reward sample in the dataset. The model begins compressing its reasoning (1,754 → 810 chars by step 70). Format compliance reaches 1.0 at step 40 after a prolonged learning period. Step 70 is notably weak (reward 0.529, bond score 0.631, FE score 0.308)—a local trough between formation of the compression heuristic and its refinement.

3. **Compressed Heuristics (Steps 80–100):** Reasoning has been compressed to <750 chars average. The model produces minimal preambles and proceeds directly to CIF output. Surprisingly, step 80 achieves the best mean reward (0.766) with some of the shortest reasoning. But the final checkpoint (step 100) shows degradation: reward drops to 0.520, bond scores collapse to 0.670, and formation energy falls to 0.283.

### 6.2 Reasoning Compression vs. Reasoning Elaboration

The most scientifically interesting finding is the **divergent reasoning strategies** between the 30B and 230B models:

| Property | 30B Model | 230B Model |
|----------|-----------|------------|
| Reasoning length trend | Decreasing (1,532 → 574 chars, −63%) | Increasing (3,966 → 7,320 chars, +85%) |
| Peak reasoning step | Step 30 (1,754 chars) | Step 70 (7,320 chars) |
| Bond keyword trend | 2.0 → 0.0 (eliminated) | 1.6 → 2.4 (growing) |
| Symmetry keyword trend | 0.6 → 0.0 (eliminated) | 6.4 → 6.9 at step 70 (sustained) |
| Wyckoff/prototype mentions | Never | Emerging by step 40+ |
| Peak reward | 0.766 (step 80) | 0.954 (step 60) |
| FE score at peak | 0.658 | 0.896 (step 70: 0.896, step 60: 0.957) |

This pattern suggests two fundamentally different RL optimization strategies mediated by model capacity:

- **230B model:** Has sufficient capacity to develop and maintain elaborate domain-specific reasoning. RL rewards the model for deeper structural analysis, and the model can represent the mapping from detailed reasoning to better structures.

- **30B model:** Lacks the capacity to maintain elaborate reasoning while also generating correct CIF structures. RL instead selects for **reasoning compression**—the model learns that short, formulaic preambles suffice, and allocates its limited capacity toward the structural output itself.

This is analogous to the distinction between **System 2** (deliberative, step-by-step) and **System 1** (fast, intuitive) reasoning in cognitive science. The 230B model develops System 2 reasoning; the 30B model converges on System 1.

### 6.3 The Conservatism Trap

Both models learn to make increasingly conservative lattice perturbations, but the 30B model takes this to an extreme:

| Step | 30B Δa (Å) | 230B Δa (Å) |
|------|-------------|-------------|
| 0    | 0.025       | 0.053       |
| 70   | 0.008       | 0.084       |
| Final| 0.002 (step 90) | 0.017 (step 60) |
| Reduction | 91% | 68% |

By step 90, the 30B model modifies the `a` lattice parameter by only 0.002 Å on average—essentially returning the input structure unchanged. This extreme conservatism is a **safe but limited strategy**: it avoids catastrophic bond-length violations (hence reasonable bond scores of ~0.82) but cannot correct genuinely misplaced lattice parameters (hence persistently high formation energies of ~1.1–1.9 eV/atom).

### 6.4 Formation Energy: The Capacity Bottleneck

The formation energy component reveals the starkest capacity difference:

- **30B model:** Mean FE oscillates between 0.93 and 1.86 eV/atom across training. The percentage of samples below threshold never exceeds 33%. Formation energy score peaks at 0.658.
- **230B model:** FE score reaches 0.957 at step 60 and sustains 0.896 at step 70. The larger model can generate structures that approach thermodynamic stability.

The 30B model's formation energies are 10–20× above the 0.1 eV/atom threshold. The model has learned to format valid CIF files, preserve composition, and produce reasonable bond lengths—but genuine crystal-structure relaxation (energy minimization) appears to require either greater model capacity or longer training.

### 6.5 Catastrophic Failures

The 30B model exhibits failure modes not seen in the 230B model:

- **Zero-reward sample** at step 30 (SnF₃): The model recognizes a symmetry mismatch but cannot produce a valid output.
- **Formation energy spikes** exceeding 5 eV/atom: Steps 30 (5.653 eV) and 60 (5.467 eV) contain samples with extremely high formation energies, indicating catastrophic misplacement of atoms.
- **Late-stage regression:** Step 100 shows degradation in band score (0.670, lowest since step 0) and formation energy (0.283), suggesting possible training instability or overfitting to the compression heuristic.

### 6.6 Limitations

- **Sample size:** 87 total samples (7–8 per step) limits statistical power, though this is larger than the 230B dataset (63 samples).
- **Confounded compositions:** Different compositions appear at different steps, making it difficult to disentangle compositional difficulty from training progress.
- **Reward sparsity:** The formation energy reward component provides a gradient (linear scoring), but the actual FE distance from threshold is large enough that the gradient signal may be weak.

---

## 7. Conclusion

Reinforcement learning on the crystal-structure relaxation task produces qualitatively different learning dynamics in the 30B model compared to the 230B model:

1. **Format and composition** are mastered, but slowly: format compliance requires ~40 steps (vs. 10 for the 230B model).

2. **Reasoning compresses** rather than elaborates: the 30B model's thinking blocks shrink by 63% over training, eliminating domain-specific vocabulary (bond lengths, symmetry, coordination) in favor of minimal procedural preambles. This is the opposite of the 230B model's 85% reasoning growth (through step 70).

3. **Physical conservatism is extreme:** Lattice perturbations shrink by 91%, essentially converging on a "pass-through" strategy that returns near-identical structures.

4. **Formation energy remains unsolved:** Mean FE values stay 10–20× above threshold throughout training. The smaller model lacks the capacity to perform genuine structure optimization.

5. **Reward is non-monotonic and oscillatory:** The best checkpoint (step 80, reward 0.766) is 20% below the 230B model's peak (0.954), and the final checkpoint (step 100) shows regression.

The 30B model's learning trajectory suggests that **model scale is a critical determinant of RL-induced reasoning quality** for scientific tasks. Below a capacity threshold, RL optimization selects for reasoning compression (efficient but shallow heuristics) rather than reasoning elaboration (rich domain modeling). The 30B model becomes a competent CIF formatter and cautious structure perturber, but does not develop the crystallographic insight that emerges in the 230B model.

---

## Appendix: Reproduction

All analyses can be reproduced from the workspace:
```bash
# Core analysis
python3 analyze_30B_rollouts.py > analysis_output.txt

# Formation energy extraction
python3 extract_fe_scores_30B.py

# Interactive exploration
jupyter notebook training_run_analysis.ipynb
```

Data files: `all_samples.csv`, `all_samples.jsonl`
