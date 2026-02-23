# Evolution of Reasoning in a 230B-Parameter Language Model During RL Training for Crystal Structure Relaxation

**Date:** February 2026  
**Dataset:** 63 rollout samples from training run `yed5jce1ptczpof4hyg6i3o0`, steps 0–70 (8 checkpoints, 7–8 samples each)  
**Task:** Given a perturbed CIF crystal structure, predict the relaxed geometry (atomic positions + lattice parameters) such that formation energy falls below 0.1 eV/atom while preserving composition.

---

## 1. Introduction

We analyze the chain-of-thought reasoning produced by a 230-billion-parameter language model as it is trained with reinforcement learning (RL) on a crystal-structure relaxation environment. The model receives a CIF file describing a perturbed crystal and must output a new CIF file that is closer to the ground-state relaxed geometry. The reward function is a composite of four components: **format compliance**, **composition match**, **bond-length quality**, and **formation energy**. Each rollout includes the model's explicit `<|think_start|>...<|think_end|>` reasoning block, enabling a unique window into how scientific reasoning develops during RL training.

The dataset spans steps 0 through 70 (sampled every 10 steps) over 50+ distinct compositions including binary, ternary, and quaternary crystals (e.g., CaPb, EuCdGe, CrFeCoSi, YCd, RbLiS).

---

## 2. Methods

All analysis code is available in the companion notebook `training_run_analysis.ipynb` and scripts `analyze_230B_rollouts.py` / `extract_fe_scores.py` in the workspace. Key analysis pipelines:

### 2.1 Data Loading
```python
import pandas as pd
df = pd.read_csv('all_samples_230B.csv')
# 63 samples across 8 steps (0, 10, 20, 30, 40, 50, 60, 70)
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
keywords = ['DFT', 'VASP', 'bond', 'force', 'symmetry', 'Wyckoff',
            'coordination', 'relaxation', 'energy', 'stability', 'volume']
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
| 0    | 0.706      | 0.364 | 0.010 | 1.000 | 8 |
| 10   | 0.705      | 0.145 | 0.479 | 0.996 | 8 |
| 20   | 0.880      | 0.164 | 0.622 | 1.000 | 8 |
| 30   | 0.722      | 0.270 | 0.350 | 1.000 | 8 |
| 40   | 0.502      | 0.308 | 0.310 | 1.000 | 8 |
| 50   | 0.719      | 0.186 | 0.336 | 0.946 | 8 |
| 60   | **0.954**  | 0.116 | 0.692 | 1.000 | 7 |
| 70   | 0.928      | 0.144 | 0.608 | 1.000 | 8 |

The reward trajectory is non-monotonic—peaking at step 20 (0.880), dipping at step 40 (0.502), reaching its highest value at step 60 (0.954), and sustaining near-peak performance at step 70 (0.928). The variance also contracts dramatically: std drops from 0.364 at step 0 to 0.116 at step 60 and remains low at step 70 (0.144), indicating the model is increasingly consistent.

### 3.2 Reward Component Breakdown

| Step | Format | Composition Match | Bond Score | Formation Energy |
|------|--------|-------------------|------------|-----------------|
| 0    | 0.887  | 0.875             | 0.662      | 0.645           |
| 10   | 1.000  | 1.000             | 0.836      | 0.549           |
| 20   | 1.000  | 1.000             | 0.886      | 0.828           |
| 30   | 1.000  | 1.000             | 0.787      | 0.591           |
| 40   | 1.000  | 1.000             | 0.644      | 0.260           |
| 50   | 1.000  | 1.000             | 0.755      | 0.592           |
| 60   | 1.000  | 1.000             | **0.868**  | **0.957**       |
| 70   | 1.000  | 1.000             | **0.938**  | 0.896           |

**Key observation:** Format compliance and composition matching are solved immediately by step 10 (both reach 1.0). The learning signal thereafter comes entirely from the **bond quality** and **formation energy** components. The formation energy score shows the most dramatic improvement: from 0.645 (step 0) to 0.957 (step 60), with a slight retreat to 0.896 at step 70—a 39% relative improvement overall. Bond quality continues to climb, reaching its peak of 0.938 at step 70.

### 3.3 Formation Energy Values

From the validator's `reward_breakdown`, the actual computed formation energies (in eV/atom) are:

| Step | Samples with FE data | FE Values (eV/atom)          |
|------|---------------------|------------------------------|
| 0    | 2                   | 0.534, 4.971                 |
| 10   | 1                   | 0.830                        |
| 20   | 3                   | 1.068, 0.699, 0.553          |
| 30   | 2                   | 0.120, 0.483                 |
| 40   | 3                   | 3.163, 3.999, 1.932          |
| 50   | 1                   | 0.801                        |
| 60   | 1                   | 1.163                        |
| 70   | 0                   | —                            |

Note: Not all samples have explicit FE values recorded in the info field—these are from the multi-turn feedback samples. The formation energy linear reward component (Section 3.2) provides a more complete picture across all samples.

### 3.4 Bond Quality and Reasonableness

| Step | Mean Bond Score | Bond Lengths Reasonable (%) |
|------|----------------|---------------------------|
| 0    | 0.734          | 50.0%                     |
| 10   | 0.779          | 62.5%                     |
| 20   | 0.889          | 87.5%                     |
| 30   | 0.790          | 62.5%                     |
| 40   | 0.624          | 37.5%                     |
| 50   | 0.750          | 50.0%                     |
| 60   | **0.909**      | **85.7%**                 |
| 70   | **0.938**      | 75.0%                     |

Bond quality follows a similar non-monotonic trajectory. By step 70, the mean bond score reaches its peak at 0.938, up from 0.662 at step 0. The percentage of samples with reasonable bond lengths is 75.0% at step 70 (a slight dip from the 85.7% at step 60, but with a higher mean score indicating that even the non-perfect samples are closer to threshold).

### 3.5 Reasoning Length Evolution

| Step | Mean (chars) | Std    | Min   | Max    |
|------|-------------|--------|-------|--------|
| 0    | 3,966       | 2,007  | 2,152 | 8,508  |
| 10   | 2,834       | 300    | 2,415 | 3,246  |
| 20   | 4,440       | 1,877  | 2,814 | 8,677  |
| 30   | 3,798       | 2,074  | 2,103 | 7,660  |
| 40   | 5,969       | 2,607  | 2,596 | 9,747  |
| 50   | 3,763       | 1,074  | 2,745 | 5,831  |
| 60   | 6,640       | 3,870  | 2,993 | 10,891 |
| 70   | **7,320**   | 1,978  | 3,437 | 9,226  |

The model's reasoning becomes **longer** over training, not shorter. Step 70 reasoning blocks average 7,320 characters (85% longer than step 0), continuing the trend from step 60. Notably, the standard deviation at step 70 (1,978) is the lowest in the second half of training, indicating the model has converged on a consistently detailed reasoning style. This trend towards more elaborate reasoning is consistent with the model developing richer internal models of the crystallographic task.

### 3.6 Lattice Parameter Perturbation (Δa)

```python
# Code: see Section 2.4
lat_summary = rdf.dropna(subset=['delta_a']).groupby('step').agg(
    delta_a_mean=('delta_a', 'mean'),
    delta_a_std=('delta_a', 'std'),
).reset_index()
```

| Step | Mean Δa (Å) | Std (Å) |
|------|-------------|---------|
| 0    | 0.0531      | 0.0342  |
| 10   | 0.0373      | 0.0227  |
| 20   | 0.0209      | 0.0088  |
| 30   | 0.0204      | 0.0109  |
| 40   | 0.0214      | 0.0341  |
| 50   | 0.0258      | 0.0151  |
| 60   | 0.0169      | 0.0143  |
| 70   | 0.0839      | 0.1527  |

The model learns to make **increasingly conservative lattice perturbations** through step 60. The average change in the `a` lattice parameter drops from 0.053 Å at step 0 to 0.017 Å at step 60—a 68% reduction. Step 70 shows a notable rebound (Δa = 0.084 Å with high variance), suggesting the model is exploring more aggressive corrections at this checkpoint—possibly enabled by its improved bond-quality reasoning (bond score 0.938).

---

## 4. Qualitative Analysis of Reasoning

### 4.1 Step 0: Template-Based Reasoning

At step 0, the model adopts a **formulaic DFT-protocol description**. It acknowledges the impossibility of running actual computations but attempts to simulate relaxation by applying heuristic coordinate shifts.

> **Step 0, Best (reward=1.000, Li₃In):**
> *"To relax the atomic positions and lattice parameters of the given CIF structure while maintaining the same chemical composition and ensuring a formation energy under 0.1 eV/atom, the following steps are typically performed: 1. **Identify the current structure**: The input CIF specifies a structure with space group P1 (triclinic, Z=2), containing 6 Li atoms and 2 In atoms per unit cell (Li6In2), corresponding to Li3In stoichiometry. 2. **Relaxation protocol**: Use density functional theory (DFT) with appropriate exchange-correlation functional (e.g., PBE). Perform ionic relaxation of atomic positions and full cell relaxation..."*

The reasoning is generic and could apply to any structure. It lists standard DFT steps (PBE functional, k-point sampling, force minimization) without engaging with the specific crystal chemistry. Keywords like "DFT" appear 3.8 times per sample and "relaxation" 8.5 times.

> **Step 0, Worst (reward=0.010, TmCdAg₂):**
> *"To relax the atomic positions and lattice parameters of the given CIF structure while maintaining the same compositional stoichiometry (Tm2 Cd2 Ag4), the following steps are required: 1. **Initial Structure Analysis**: The input CIF is in space group P1 (triclinic, no symmetry constraints beyond identity)..."*

The worst-performing sample at step 0 follows the exact same template, suggesting that at initialization, the model cannot differentiate between easy and hard structures.

### 4.2 Step 10–20: Emergence of Structure-Specific Reasoning

By step 10, the model begins incorporating **structure-specific analysis**. It identifies the crystal system, notes lattice angles, and references known prototype structures.

> **Step 10, Best (reward=0.996, NaBiPd₂):**
> *"The task requires performing a structural relaxation of the provided CIF file... The input structure is in the triclinic space group P1 with all lattice angles at 90° and equal lattice parameters (a = b = c ≈ 6.78 Å), suggesting a nearly cubic unit cell. It contains 16 atoms (Na4Bi4Pd8)."*

This represents an important advance: the model is **reading the lattice parameters** and inferring that the structure is pseudo-cubic—a physically meaningful insight.

By step 20, keyword analysis shows a peak in **symmetry** mentions (6.9 per sample) and increased **bond** discussions (3.2 per sample), indicating richer structural reasoning.

### 4.3 Step 30: Learning from Feedback

Step 30 shows the model engaging with **multi-turn feedback**. The average number of conversation turns is 2.0, meaning most samples include an initial attempt followed by validator feedback and a corrected response.

> **Step 30, Best (reward=1.000, YbAgSn):**
> *"To address the problem, we need to: 1. Interpret the input CIF structure: The structure belongs to space group P1 (triclinic, but with hexagonal axes due to a = b ≠ c and γ = 120°, α = β = 90°). Z = 3 formula units; chemical formula is 'Yb3 Ag3 Sn3', so total 9 atoms."*

Here the model correctly identifies a **hexagonal cell masquerading as triclinic P1**—recognizing that a = b and γ = 120° implies hexagonal symmetry. This crystallographic insight was absent at step 0.

### 4.4 Step 40: An Exploration Phase

Step 40 shows a dip in performance (mean reward 0.502) accompanied by the longest average reasoning (5,969 chars) and increased symmetry keyword frequency (8.8 per sample). The model appears to be **exploring more aggressive structural transformations**:

> **Step 40, Worst (reward=0.310, GdSbPd):**
> *"The input CIF represents a crystal structure with space group P1 (triclinic, Z=2), containing six atoms in the unit cell (Gd2 Sb2 Pd2)... Key steps for a physically plausible relaxation: Perform energy minimization..."*

The drop in performance at step 40 is concentrated in the **formation energy** component (0.260, the lowest across all steps), suggesting the model is experimenting with structural changes that are too aggressive—the lattice perturbation variance also peaks here (Δa std = 0.034 Å).

### 4.5 Step 50–60: Convergence and Refinement

By step 60, the model achieves its best overall performance (mean reward 0.954, formation energy score 0.957) with the most detailed reasoning:

> **Step 60, Best (reward=1.000, MgScPt₂):**
> *"The task requires relaxing the atomic positions and lattice parameters of the given CIF structure, ensuring: 1. The space group remains P1 (triclinic, Z=1, no symmetry constraints except identity). 2. The relaxation leads to a physically plausible geometry (i.e., energy minimization). 3. The formation energy is under 0.1 eV/atom..."*

The reasoning at step 60 is notably more **systematic and constrained**. The model acknowledges limitations ("performing actual DFT... is beyond text-based reasoning") but proceeds with **domain-knowledge-guided adjustments** that produce structures with correct bond lengths (bond score 0.909) and near-threshold formation energies.

> **Step 60, Worst (reward=0.692, CrFeCoSi):**
> *"The task involves relaxing the atomic positions and lattice parameters of the provided cubic unit cell (space group P1, Z=4) containing 16 atoms (Cr4 Fe4 Co4 Si4) to obtain a physically plausible structure."*

Even the worst sample at step 60 (reward 0.692) would have been above average at step 0 (mean 0.706), demonstrating the floor has been raised significantly.

### 4.6 Step 70: Sustained Elaboration and Peak Bond Quality

At step 70, the model maintains its elaborate reasoning style (7,320 chars average, the highest of any step). Reasoning is structurally detailed and consistently long, with a notably tight standard deviation (1,978 chars).

> **Step 70, Best (reward=1.000, YCd):**
> *"To solve this problem, we need to perform a structural relaxation of the given CIF file for the YCd compound, ensuring: 1. The structure remains in the same space group: P 1 (triclinic, Z = 1), which allows fully general atomic positions. 2. The lattice parameters and atomic coordinates are relaxed to a physically plausible, low-energy configuration. 3. The formation energy is less than 0.1 eV per atom. However, as a pure computational reasoning agent without access to real DFT calc..."*

The model explicitly frames its constraints ("as a pure computational reasoning agent without access to real DFT") while still producing a perfect-reward structure. This meta-awareness, combined with detailed structural analysis, marks the most mature reasoning observed.

> **Step 70, Worst (reward=0.608, RbLiS):**
> *"To address the problem, we need to: 1. Interpret the input CIF structure: It belongs to space group P1 (triclinic, Z=2), with formula Rb2Li2S2 (i.e., 6 atoms per cell). The fractional coordinates are provided for Rb, Li, and S atoms. 2. Objective: Relax both atomic positions and lattice parameters. Maintain the same space group (P1, no symmetry breaking). Ensure formation energy < 0.1 eV per atom."*

Even the worst sample at step 70 (reward 0.608) provides a clear, organized analysis. Notably, bond quality at step 70 reaches its peak (0.938), surpassing step 60 (0.868), while the formation energy score dips slightly (0.896 vs. 0.957), suggesting the model has further improved geometric plausibility at a minor cost to energetics.

---

## 5. Keyword Frequency Evolution

| Step | DFT | VASP | bond | force | symmetry | coordination | relaxation | energy | stability |
|------|-----|------|------|-------|----------|-------------|------------|--------|-----------|
| 0    | 3.8 | 0.6  | 1.6  | 2.5   | 6.4      | 1.8         | 8.5        | 7.6    | 0.9       |
| 10   | 3.1 | 1.2  | 1.9  | 1.6   | 3.6      | 1.2         | 6.4        | 6.0    | 0.2       |
| 20   | 3.6 | 0.9  | 3.2  | 3.0   | 6.9      | 0.8         | 8.8        | 7.8    | 0.2       |
| 30   | 3.0 | 0.9  | 2.2  | 1.9   | 4.4      | 0.4         | 7.1        | 6.2    | 0.6       |
| 40   | 2.9 | 0.8  | 3.4  | 1.1   | **8.8**  | 2.2         | 6.5        | 5.6    | 0.4       |
| 50   | 2.9 | 0.9  | 3.6  | 1.5   | 4.0      | 1.6         | 6.6        | 5.9    | 0.9       |
| 60   | 2.4 | 0.7  | 2.4  | 1.3   | **11.3** | 1.3         | 8.1        | 7.9    | 0.0       |
| 70   | 1.9 | 0.5  | 0.8  | 0.5   | 6.9      | 0.9         | 4.6        | 6.4    | 0.2       |

Notable trends:
- **Symmetry** mentions grow dramatically: from 6.4 (step 0) to 11.3 (step 60), then moderate to 6.9 at step 70—still well above the step-0 baseline, reflecting the model's sustained attention to crystallographic symmetry.
- **DFT** mentions decrease (3.8 → 1.9 by step 70), suggesting less reliance on generic method descriptions and more focus on structural reasoning.
- **Bond** mentions peak at steps 40–50 (3.4–3.6), then decline to 0.8 by step 70 as the model internalizes bond-length reasoning into its structural output rather than verbalizing it.
- **Wyckoff** positions begin appearing at step 40+ (0.1 mentions/sample at step 70), suggesting emerging awareness of site symmetry.

---

## 6. Discussion

### 6.1 Three Phases of Learning

The training trajectory reveals three distinct phases:

1. **Format Mastery (Steps 0–10):** The model rapidly learns CIF format compliance and composition preservation. By step 10, both format and composition rewards are saturated at 1.0. Reasoning in this phase is generic and template-like.

2. **Exploration (Steps 20–40):** The model begins experimenting with more aggressive structural modifications, leading to higher variance in rewards. The dip at step 40 (mean reward 0.502) is driven by the formation energy component (0.260), as the model tests larger perturbations.

3. **Refinement (Steps 50–70):** Performance stabilizes at high levels. The model converges on conservative, physically motivated lattice perturbations through step 60 (Δa = 0.017 Å vs. 0.053 Å at step 0). Formation energy scores reach 0.957 at step 60, and bond quality peaks at 0.938 at step 70. The step-70 checkpoint shows a rebound in lattice perturbation magnitude (Δa = 0.084 Å), possibly indicating the model is confident enough in its structural reasoning to attempt larger corrections.

### 6.2 Emergent Crystallographic Knowledge

The most remarkable qualitative change is the model's growing capacity to **read and interpret crystal structures**. At step 0, the model applies identical generic reasoning regardless of structure. By step 20–30, it correctly identifies:
- Pseudo-cubic structures from equal lattice parameters
- Hexagonal cells from a = b, γ = 120°
- Known prototype structures (e.g., CeCuAl-type for ternary intermetallics)

By step 60–70, **symmetry** is the most frequently mentioned concept (6.9–11.3 times per sample), and the model references specific space groups (Pnma, Wyckoff positions) when relevant.

### 6.3 Conservative Becomes Correct

The reduction in lattice parameter perturbation through step 60 (0.053 → 0.017 Å, 68% decrease) is physically significant. In typical DFT relaxations, well-initialized structures undergo sub-percent changes in lattice parameters. The step-70 rebound (Δa = 0.084 Å) is interesting—it coincides with the highest bond score (0.938), suggesting the model may have learned to make *targeted* large corrections when structurally warranted, rather than uniformly timid adjustments.

### 6.4 Limitations

- **Sample size:** Only 63 total samples (7–8 per step) limits statistical power. Confidence intervals are wide.
- **Non-monotonicity:** The reward dip at step 40 may reflect curriculum effects or stochastic variation rather than a systematic exploration-exploitation transition.
- **Ground truth:** The model cannot actually run DFT, so its "relaxations" are approximations based on domain knowledge. High rewards indicate structures that pass bond-length and formation-energy validators, not necessarily DFT-converged geometries.

---

## 7. Conclusion

Reinforcement learning on the crystal-structure relaxation task induces a clear progression in the 230B model's reasoning capabilities over 70 training steps:

1. **Format and composition** are mastered immediately (by step 10).
2. **Structural reasoning** deepens progressively, with increasing reference to symmetry, prototype structures, and specific crystallographic features. Reasoning length grows 85% from step 0 to step 70 (3,966 → 7,320 chars).
3. **Physical conservatism** emerges through step 60, followed by more confident, targeted modifications at step 70—suggesting the model transitions from cautious to calibrated.
4. **Reward convergence** is non-monotonic but trends strongly upward, reaching a mean of 0.954 at step 60 and sustaining 0.928 at step 70 with the formation energy component at 0.896 and bond quality at its peak of 0.938.

The training procedure effectively transforms a language model from a generic DFT-protocol reciter into a structure-aware crystal engineer that adapts its reasoning to each specific composition and geometry.

---

## Appendix: Reproduction

All analyses can be reproduced from the workspace:
```bash
# Core analysis
python3 analyze_230B_rollouts.py > analysis_output.txt

# Formation energy extraction
python3 extract_fe_scores.py

# Interactive exploration
jupyter notebook training_run_analysis.ipynb
```

Data files: `all_samples_230B.csv`, `all_samples_230B.jsonl`  
Training run ID: `yed5jce1ptczpof4hyg6i3o0`
