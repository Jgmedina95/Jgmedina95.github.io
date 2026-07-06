# Crystal Relaxation RLM Transition Report

## Purpose

This report summarizes the local eval artifacts for the legacy
`crystal_relaxation` environment and the RLM-native
`crystal-relaxation-rlm` environment. The goal is to track the transition from
the classic short multi-turn setup to the sandbox-backed RLM harness, identify
which models are useful baselines, and record the next full-eval configuration.

## Source Artifacts

Automatic uploads were initially skipped when local environments were ahead of
their Hub copies, so most original `metadata.json` files do not contain a
platform eval id. The runs below were later pushed manually; the durable local
eval id is the 8-character result directory and the platform eval id is the Hub
identifier returned by `prime eval push`.

## Hub Eval IDs

| Environment | Local eval id | Platform eval id | Note |
|---|---|---|---|
| `crystal_relaxation` | `8962a2f4` | `gnlsgjic0fzv8bgi3sj6gmnv` | Classic GPT-5.5 best run. |
| `crystal_relaxation` | `16d858ab` | `f35xhbpdyasd8rujc4yz492y` | Classic GPT-5.5 3-turn rerun. |
| `crystal_relaxation` | `536b00ba` | `mv0w6mhdf8v2l7849hlt4v7n` | Classic Gemini baseline. |
| `crystal_relaxation` | `4427177a` | `k91edvkw3uckjujj377v795v` | Pushed from sanitized temp copy; one non-finite reward/formation-energy value was replaced for Hub numeric validation. Original local artifact unchanged. |
| `crystal_relaxation` | `1da43e7e` | `qitlcfvjug1ic8uwa3i8nn26` | Classic GPT-5.4-mini baseline. |
| `crystal_relaxation` | `b7aee56f` | `p3kl3a7soe3t6h606updizl4` | Classic GPT-5.4-mini 3-turn rerun. |
| `crystal_relaxation` | `f2c5f410` | `u1z0hpvcemwh4yuyw5qn8r4o` | Classic Qwen run. |
| `crystal_relaxation` | `83a9b09c` | `w7q93igg6ecb04nix9xm4roy` | Classic Claude baseline. |
| `crystal_relaxation` | `32315f0c` | `sack3pc9yw9rtqvdvs1h9fw5` | Classic Claude 3-turn rerun. |
| `crystal-relaxation-rlm` | `45491238` | `uny28v06ec1c3x5ehty73k7x` | RLM GPT-5.5 old reference. |
| `crystal-relaxation-rlm` | `003b7c4d` | `brmn3i8lpiok8c3jjlkyazgr` | RLM GPT-5.5 trusted 10-turn, 3-check run. |
| `crystal-relaxation-rlm` | `34ff5575` | `bnpfw8cghaoqgef0kzh21y8u` | RLM GPT-5.5 non-trusted 10-turn, 3-check run. |
| `crystal-relaxation-rlm` | `60cf0943` | `ux47n1kow0lf0wfobrirpjfs` | RLM GPT-5.5 non-trusted 15-turn, 5-check run. |
| `crystal-relaxation-rlm` | `194e350b` | `qrd63h0jkbkyn1ox4g99ym30` | RLM GPT-5.4-mini old reference. |
| `crystal-relaxation-rlm` | `870af5d2` | `qyxnrab06xbmn2m6snfha3pm` | RLM GPT-5.4-mini trusted 10-turn, 3-check run. |
| `crystal-relaxation-rlm` | `0d02feeb` | `yblylsf34h3wrlrh8mfq95mu` | RLM GPT-5.4-mini non-trusted 10-turn, 3-check run. |
| `crystal-relaxation-rlm` | `5dd698c6` | `ryqrserwltmzmxeds2sg685i` | RLM GPT-5.4-mini non-trusted 15-turn, 5-check run. |
| `crystal-relaxation-rlm` | `2ec0705d` | `r5j0594wwfovld2sq9ngjl72` | RLM Gemini old reference. |
| `crystal-relaxation-rlm` | `e59f5b1d` | `fhr888t4ihsgjtrgs3wis3xi` | RLM Gemini trusted 10-turn, 3-check run. |
| `crystal-relaxation-rlm` | `1476c619` | `yq06vjmiys6xnvv4ghmp64wt` | RLM Gemini non-trusted 10-turn, 3-check run. |
| `crystal-relaxation-rlm` | `65760884` | `kp77ab85ixsd14stk6fbfaaa` | RLM Gemini non-trusted 15-turn, 5-check run. |
| `crystal-relaxation-rlm` | `23269e2e` | `h6wmixa8rm8hubuihd6vtg07` | RLM Claude old reference. |
| `crystal-relaxation-rlm` | `d71cef5b` | `qk5z3fjgzleudu59ccawwghg` | RLM Claude trusted 10-turn, 3-check run. |
| `crystal-relaxation-rlm` | `01cf1c10` | `sfur1hqc432i8rtu3p89xs6h` | RLM Claude non-trusted 10-turn, 3-check run. |
| `crystal-relaxation-rlm` | `786ad9af` | `ggexo49jc4af7tqakzrwl16t` | RLM Claude non-trusted 15-turn, 5-check run. |
| `crystal_relaxation_rlm` | `eb656941` | `wo3hdujv6oes83r02v3knd5a` | RLM Qwen run. |

An earlier attempt to push `4427177a` created partial Hub eval
`tl1q5pg4wujuqdpyi9iyyhr5` before sample validation rejected the non-finite
reward fields. The finalized Hub copy is `k91edvkw3uckjujj377v795v`.

- Classic eval summaries:
  - `environments/crystal_relaxation/outputs/evals/crystal_relaxation--google--gemini-2.5-flash/536b00ba/summary.md`
  - `environments/crystal_relaxation/outputs/evals/crystal_relaxation--anthropic--claude-haiku-4.5/83a9b09c/summary.md`
- RLM eval summaries:
  - `environments/crystal_relaxation_rlm/outputs/evals/crystal-relaxation-rlm--openai--gpt-5.5/45491238/summary.md`
  - `environments/crystal_relaxation_rlm/outputs/evals/crystal-relaxation-rlm--openai--gpt-5.4-mini/194e350b/summary.md`
  - `environments/crystal_relaxation_rlm/outputs/evals/crystal-relaxation-rlm--google--gemini-2.5-flash/2ec0705d/summary.md`
  - `environments/crystal_relaxation_rlm/outputs/evals/crystal-relaxation-rlm--google--gemini-2.5-flash/a63f0de7/summary.md`
  - `environments/crystal_relaxation_rlm/outputs/evals/crystal-relaxation-rlm--anthropic--claude-haiku-4.5/23269e2e/summary.md`
  - `environments/crystal_relaxation_rlm/outputs/evals/crystal-relaxation-rlm--anthropic--claude-haiku-4.5/303e0cca/summary.md`
- Full RLM rerun metadata, 25x3 with `max_turns=10`, `max_checks=3`,
  `max_concurrent=3`:
  - `environments/crystal_relaxation_rlm/outputs/evals/crystal-relaxation-rlm--openai--gpt-5.5/003b7c4d/metadata.json`
  - `environments/crystal_relaxation_rlm/outputs/evals/crystal-relaxation-rlm--openai--gpt-5.4-mini/870af5d2/metadata.json`
  - `environments/crystal_relaxation_rlm/outputs/evals/crystal-relaxation-rlm--google--gemini-2.5-flash/e59f5b1d/metadata.json`
  - `environments/crystal_relaxation_rlm/outputs/evals/crystal-relaxation-rlm--anthropic--claude-haiku-4.5/d71cef5b/metadata.json`
- GPT-5.5 non-trusted 10-turn rerun metadata, 25x3 with `max_turns=10`,
  `max_checks=3`, `max_concurrent=5`:
  - `environments/crystal_relaxation_rlm/outputs/evals/crystal-relaxation-rlm--openai--gpt-5.5/34ff5575/metadata.json`
  - `environments/crystal_relaxation_rlm/outputs/evals/crystal-relaxation-rlm--openai--gpt-5.5/34ff5575/results.jsonl`
- GPT-5.5 non-trusted 15-turn rerun metadata, 25x3 with `max_turns=15`,
  `max_checks=5`, `max_concurrent=5`:
  - `environments/crystal_relaxation_rlm/outputs/evals/crystal-relaxation-rlm--openai--gpt-5.5/60cf0943/metadata.json`
  - `environments/crystal_relaxation_rlm/outputs/evals/crystal-relaxation-rlm--openai--gpt-5.5/60cf0943/results.jsonl`
- Cross-model non-trusted 10-turn rerun metadata, 25x3 with `max_turns=10`,
  `max_checks=3`, `max_concurrent=5`:
  - `environments/crystal_relaxation_rlm/outputs/evals/crystal-relaxation-rlm--openai--gpt-5.5/34ff5575/metadata.json`
  - `environments/crystal_relaxation_rlm/outputs/evals/crystal-relaxation-rlm--openai--gpt-5.4-mini/0d02feeb/metadata.json`
  - `environments/crystal_relaxation_rlm/outputs/evals/crystal-relaxation-rlm--google--gemini-2.5-flash/1476c619/metadata.json`
  - `environments/crystal_relaxation_rlm/outputs/evals/crystal-relaxation-rlm--anthropic--claude-haiku-4.5/01cf1c10/metadata.json`
- Cross-model non-trusted 15-turn rerun metadata, 25x3 with `max_turns=15`,
  `max_checks=5`, `max_concurrent=5`:
  - `environments/crystal_relaxation_rlm/outputs/evals/crystal-relaxation-rlm--openai--gpt-5.5/60cf0943/metadata.json`
  - `environments/crystal_relaxation_rlm/outputs/evals/crystal-relaxation-rlm--openai--gpt-5.4-mini/5dd698c6/metadata.json`
  - `environments/crystal_relaxation_rlm/outputs/evals/crystal-relaxation-rlm--google--gemini-2.5-flash/65760884/metadata.json`
  - `environments/crystal_relaxation_rlm/outputs/evals/crystal-relaxation-rlm--anthropic--claude-haiku-4.5/786ad9af/metadata.json`
- Classic 3-turn rerun metadata, 25x3 with `max_turns=3`,
  `max_concurrent=5`, real CHGNet scoring:
  - `environments/crystal_relaxation/outputs/evals/crystal_relaxation--openai--gpt-5.5/16d858ab/metadata.json`
  - `environments/crystal_relaxation/outputs/evals/crystal_relaxation--openai--gpt-5.4-mini/b7aee56f/metadata.json`
  - `environments/crystal_relaxation/outputs/evals/crystal_relaxation--google--gemini-2.5-flash/4427177a/metadata.json`
  - `environments/crystal_relaxation/outputs/evals/crystal_relaxation--anthropic--claude-haiku-4.5/32315f0c/metadata.json`
- RLM setup and scaling notes:
  - `environments/crystal_relaxation_rlm/experiments/setup_speed_report_20260511.md`

## Headline Result

The classic `crystal_relaxation` environment is still the stronger short-run
eval surface. Its corrected `1.1.5` runs fixed concurrent multi-turn state
isolation and removed target-composition leakage; corrected 25x3 reruns had
`0/75` prompt/answer/feedback target-composition mismatches.

The RLM-native `crystal-relaxation-rlm` environment is harder and slower, but
the best RLM baseline is now strong: `openai/gpt-5.5` reached `0.783` mean
reward with `55/75` full-credit rollouts under the 25x3, 15-turn, 5-check,
non-trusted-image setting. The cross-model non-trusted sweep shows that the
extra `15/5` budget helps GPT-5.5 substantially, helps Claude meaningfully,
helps GPT-5.4-mini only modestly, and hurts Gemini in this run.

## Comparable Runs

| Environment | Local eval id | Model | Run shape | Mean reward | pass@1 | pass@2 | Notes |
|---|---|---|---:|---:|---:|---:|---|
| `crystal_relaxation` | `8962a2f4` | `openai/gpt-5.5` | 25x3 | `0.998` | `1.000` | `1.000` | Best overall classic run. |
| `crystal_relaxation` | `16d858ab` | `openai/gpt-5.5` | 25x3, 3 turns | `0.758` | `0.760` | `0.880` | Strong but had `32%` model/API errors. |
| `crystal_relaxation` | `536b00ba` | `google/gemini-2.5-flash` | 25x3 | `0.687` | `0.747` | `0.933` | Strong classic baseline; `15/75` perfect rollouts. |
| `crystal_relaxation` | `4427177a` | `google/gemini-2.5-flash` | 25x3, 3 turns | `0.525` finite-only | `0.413` | `0.600` | One invalid/NaN rollout made saved mean null. |
| `crystal_relaxation` | `1da43e7e` | `openai/gpt-5.4-mini` | 25x3 | `0.624` | `0.587` | `0.787` | Solid classic baseline. |
| `crystal_relaxation` | `b7aee56f` | `openai/gpt-5.4-mini` | 25x3, 3 turns | `0.508` | `0.400` | `0.547` | Worse than the prior 2-turn classic run. |
| `crystal_relaxation` | `f2c5f410` | `qwen/qwen3-8b` | 10x3 | `0.332` | `0.267` | `0.433` | Weak classic run. |
| `crystal_relaxation` | `83a9b09c` | `anthropic/claude-haiku-4.5` | 25x3 | `0.185` | `0.200` | `0.333` | Weak; `54/75` zero-reward trajectories. |
| `crystal_relaxation` | `32315f0c` | `anthropic/claude-haiku-4.5` | 25x3, 3 turns | `0.298` | `0.267` | `0.413` | Improved over prior 2-turn classic run. |
| `crystal-relaxation-rlm` | `45491238` | `openai/gpt-5.5` | 25x3 | `0.618` | `0.493` | `0.587` | Best RLM run; `37/75` full-credit rollouts. |
| `crystal-relaxation-rlm` | `003b7c4d` | `openai/gpt-5.5` | 25x3, 10 turns, 3 checks | `0.341` | `0.240` | `0.347` | Full rerun; lower than 5-turn/1-check baseline. |
| `crystal-relaxation-rlm` | `34ff5575` | `openai/gpt-5.5` | 25x3, 10 turns, 3 checks, non-trusted image, concurrency 5 | `0.516` | `0.440` | `0.520` | Clean dependency path; `33/75` full-credit rollouts. |
| `crystal-relaxation-rlm` | `60cf0943` | `openai/gpt-5.5` | 25x3, 15 turns, 5 checks, non-trusted image, concurrency 5 | `0.783` | `0.733` | `0.760` | Best RLM run so far; `55/75` full-credit rollouts despite `7/75` sandbox-ready failures. |
| `crystal-relaxation-rlm` | `194e350b` | `openai/gpt-5.4-mini` | 25x3 | `0.348` | `0.013` | `0.027` | Good format/composition, weak energy/symmetry. |
| `crystal-relaxation-rlm` | `870af5d2` | `openai/gpt-5.4-mini` | 25x3, 10 turns, 3 checks | `0.314` | `0.013` | `0.027` | More checks improved composition but not final energy. |
| `crystal-relaxation-rlm` | `0d02feeb` | `openai/gpt-5.4-mini` | 25x3, 10 turns, 3 checks, non-trusted image, concurrency 5 | `0.376` | `0.080` | `0.120` | Reliable but weak on formation energy. |
| `crystal-relaxation-rlm` | `5dd698c6` | `openai/gpt-5.4-mini` | 25x3, 15 turns, 5 checks, non-trusted image, concurrency 5 | `0.408` | `0.107` | `0.173` | Modest gain from larger budget; `8/75` full-credit rollouts. |
| `crystal-relaxation-rlm` | `2ec0705d` | `google/gemini-2.5-flash` | 25x3 | `0.322` | `0.160` | `0.280` | Viable after lowering concurrency; `10/75` full-credit rollouts. |
| `crystal-relaxation-rlm` | `e59f5b1d` | `google/gemini-2.5-flash` | 25x3, 10 turns, 3 checks | `0.252` | `0.093` | `0.173` | Completed with `3/75` model/agent errors. |
| `crystal-relaxation-rlm` | `1476c619` | `google/gemini-2.5-flash` | 25x3, 10 turns, 3 checks, non-trusted image, concurrency 5 | `0.297` | `0.160` | `0.280` | Similar pass rate to older RLM baseline, with more low-reward partials. |
| `crystal-relaxation-rlm` | `65760884` | `google/gemini-2.5-flash` | 25x3, 15 turns, 5 checks, non-trusted image, concurrency 5 | `0.259` | `0.107` | `0.173` | Regressed under larger budget; `4%` error rate. |
| `crystal-relaxation-rlm` | `23269e2e` | `anthropic/claude-haiku-4.5` | 25x3 | `0.320` | `0.067` | `0.133` | Viable after lowering concurrency; physical checks remain weak. |
| `crystal-relaxation-rlm` | `d71cef5b` | `anthropic/claude-haiku-4.5` | 25x3, 10 turns, 3 checks | `0.104` | `0.000` | `0.000` | Degenerated to format-only behavior. |
| `crystal-relaxation-rlm` | `01cf1c10` | `anthropic/claude-haiku-4.5` | 25x3, 10 turns, 3 checks, non-trusted image, concurrency 5 | `0.303` | `0.133` | `0.200` | Stable but often hits the turn limit. |
| `crystal-relaxation-rlm` | `786ad9af` | `anthropic/claude-haiku-4.5` | 25x3, 15 turns, 5 checks, non-trusted image, concurrency 5 | `0.454` | `0.267` | `0.413` | Best non-GPT-5.5 RLM larger-budget result. |
| `crystal_relaxation_rlm` | `eb656941` | `qwen/qwen3-8b` | 10x3 | `0.027` | `0.000` | `0.000` | Not useful in that run. |

## Classic Environment Findings

`openai/gpt-5.5` nearly saturates the corrected classic rubric: format,
composition, and formation-energy means are `1.000`, with bond lengths at
`0.985`. This makes it the best reference model for whether the task and rubric
are solvable.

`google/gemini-2.5-flash` is the strongest non-OpenAI classic baseline. Its
25x3 corrected run produced mean reward `0.687`, `56/75` passing trajectories,
and `15/75` perfect trajectories. The strongest component was bond-length
plausibility, followed by formation energy.

`anthropic/claude-haiku-4.5` was weak on the classic environment. It produced
only `15/75` passing trajectories, with `54/75` trajectories at exactly zero.
The recorded format component was zero for all trajectories, although nonzero
final reward was still possible through other components.

## Classic 3-Turn Rerun

The classic environment was rerun on 2026-05-19 with the same four model
families used in the RLM sweep. The run shape was `25` examples, `3` rollouts
per example, `max_turns=3`, `max_concurrent=5`, `max_tokens=8192`, the same
four rubrics (`format`, `composition`, `formation_energy`, `bond_lengths`),
`mock=false`, `testing=false`, and the local fine-tuned CHGNet checkpoint.

| Model | Local eval id | CLI job id | Mean reward | pass@1 | pass@2 | Format | Composition | Formation energy | Bond lengths | Avg turns | Errors |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `openai/gpt-5.5` | `16d858ab` | `crystal_relaxation_openai_gpt_5.5_20260519_135457_6e46496e` | `0.758` | `0.760` | `0.880` | `0.763` | `0.760` | `0.760` | `0.747` | `2.09` | `0.320` |
| `openai/gpt-5.4-mini` | `b7aee56f` | `crystal_relaxation_openai_gpt_5.4_mini_20260519_135456_d943bc44` | `0.508` | `0.400` | `0.547` | `0.763` | `0.920` | `0.333` | `0.630` | `2.97` | `0.000` |
| `google/gemini-2.5-flash` | `4427177a` | `crystal_relaxation_google_gemini_2.5_flash_20260519_135456_694c63fe` | `0.525` finite-only | `0.413` | `0.600` | `0.768` | `0.840` | `0.364` finite-only | `0.660` | `2.96` | `0.000` |
| `anthropic/claude-haiku-4.5` | `32315f0c` | `crystal_relaxation_anthropic_claude_haiku_4.5_20260519_135456_7704018c` | `0.298` | `0.267` | `0.413` | `0.000` | `0.613` | `0.218` | `0.505` | `2.99` | `0.000` |

Gemini produced one invalid numeric value in formation-energy scoring, so the
saved `metadata.json` has `avg_reward: null` and
`formation_energy_linear: null`. The finite-only mean over the other recorded
reward values is `0.525`; pass@k was still saved normally.

Compared with the latest RLM `15/5` sweep:

| Model | Classic 3-turn mean | Classic pass@1 | Classic pass@2 | RLM 15/5 mean | RLM pass@1 | RLM pass@2 | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| `openai/gpt-5.5` | `0.758` | `0.760` | `0.880` | `0.783` | `0.733` | `0.760` | Similar mean; classic has better pass@2 but many model/API errors. |
| `openai/gpt-5.4-mini` | `0.508` | `0.400` | `0.547` | `0.408` | `0.107` | `0.173` | Classic remains much easier for mini. |
| `google/gemini-2.5-flash` | `0.525` finite-only | `0.413` | `0.600` | `0.259` | `0.107` | `0.173` | Classic is far stronger, but one NaN scorer value needs investigation. |
| `anthropic/claude-haiku-4.5` | `0.298` | `0.267` | `0.413` | `0.454` | `0.267` | `0.413` | Same pass rates, but RLM has higher mean partial credit. |

The old environment still rewards the smaller models more generously than the
RLM harness, especially GPT-5.4-mini and Gemini. The exception is Claude:
under `15/5`, RLM gives much better continuous partial credit while preserving
the same pass@1 and pass@2 as the classic 3-turn run.

GPT-5.5 is the most nuanced comparison. Classic 3-turn reward (`0.758`) is
close to RLM 15/5 reward (`0.783`), and classic pass@2 is higher. However,
the classic GPT-5.5 run had `0.320` average error from
`EmptyModelResponseError` and `UnprocessableEntityError`, while the RLM 15/5
run's nonzero error tail was mostly sandbox readiness. The classic run also
stopped early on `structure_meets_criteria` in `20%` of rollouts, whereas the
RLM harness usually needs more turns and explicit check-structure calls to
reach the same endpoint.

Automatic upload was skipped for these classic evals because the local
`crystal_relaxation` environment is ahead of the Hub copy.

## RLM Environment Findings

`openai/gpt-5.5` was the strongest RLM baseline under the original 5-turn,
1-check budget:

| Metric | Value |
|---|---:|
| Mean reward | `0.618` |
| Full-credit rollouts | `37 / 75` |
| Format valid | `75 / 75` |
| Composition match | `56 / 75` |
| Bond lengths reasonable | `55 / 75` |
| Formation energy below threshold | `37 / 75` |
| Agent/runtime failures | `0 / 75` |

The main remaining RLM failure modes are composition validity, force proxy
failures, and hard symmetry targets. Many failures collapse to low discrete
rewards (`0.10`, `0.25`, or `0.40`), while successful rollouts often reach full
credit.

`openai/gpt-5.4-mini` is operationally reliable but physically weak. It
preserves format and composition well, but passed formation energy in only
`1/75` rollouts and matched target spacegroup in only `3/67` parseable private
metrics.

`google/gemini-2.5-flash` and `anthropic/claude-haiku-4.5` both had earlier
25x3 all-timeout RLM runs at requested concurrency `15`. Reruns at lower
effective concurrency completed normally. Those timeout runs should be treated
as provider/concurrency diagnostics, not intrinsic model-quality results.

## Full RLM Rerun: 10 Turns, 3 Checks

The 2026-05-18 full rerun used the same 25x3 shape for all four models, with
`max_concurrent=3`, `max_turns=10`, `max_checks=3`, `max_tokens=8192`, the
trusted sandbox image, and rubrics limited to `format`, `composition`,
`bond_lengths`, and `formation_energy`.

| Model | Local eval id | Mean reward | pass@1 | pass@2 | Format | Composition | Bond lengths | Formation energy | Avg turns | Errors |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `openai/gpt-5.5` | `003b7c4d` | `0.341` | `0.240` | `0.347` | `0.973` | `0.333` | `0.333` | `0.240` | `9.28` | `0.040` |
| `openai/gpt-5.4-mini` | `870af5d2` | `0.314` | `0.013` | `0.027` | `1.000` | `0.813` | `0.560` | `0.013` | `7.75` | `0.000` |
| `google/gemini-2.5-flash` | `e59f5b1d` | `0.252` | `0.093` | `0.173` | `0.960` | `0.347` | `0.320` | `0.093` | `5.93` | `0.040` |
| `anthropic/claude-haiku-4.5` | `d71cef5b` | `0.104` | `0.000` | `0.000` | `1.000` | `0.013` | `0.013` | `0.000` | `10.00` | `0.000` |

The trusted-image larger-budget run did not improve the original RLM baseline.
`openai/gpt-5.5` dropped from `0.618` in the 5-turn, 1-check run to `0.341` in
the 10-turn, 3-check run. Later non-trusted reruns showed that this was mostly
an interpreter/dependency-path issue rather than a general failure of larger
turn and checker budgets.

`openai/gpt-5.4-mini` is the clearest example of partial progress without final
success: it had high format and composition rates, and bond lengths passed in
`56%` of rollouts, but formation energy passed in only `1/75` rollouts. This
kept pass@1 at `0.013` despite a moderate mean reward.

`google/gemini-2.5-flash` remained viable at `max_concurrent=3`, but the
expanded budget was not enough to beat its earlier lower-budget RLM run. It
also had a small error rate (`3/75` rollouts), mostly empty or failed model
responses.

`anthropic/claude-haiku-4.5` was stable but ineffective: it consumed the full
10-turn budget on every rollout, preserved output format, and almost never
reached correct composition, bond lengths, or energy.

Automatic upload was skipped for these evals because the local environment is
ahead of `jmedina9/crystal-relaxation-rlm`; publish first if these exact local
results need to appear in the private Evaluations tab.

## GPT-5.5 Control Rerun: Old Budget, Higher Concurrency

To isolate the trusted-image/interpreter issue, GPT-5.5 was rerun with the old
budget shape: `25` examples, `3` rollouts each, `max_turns=5`, default
`max_checks=1`, no explicit `max_tokens`, and `trust_sandbox_image=False`.
The only intentional change from the old reference was higher requested
concurrency: `--max-concurrent 10`.

| Run | Local eval id | Mean reward | pass@1 | pass@2 | Format | Composition | Bond lengths | Formation energy | Avg turns |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Old reference | `45491238` | `0.618` | `0.493` | `0.587` | `1.000` | `0.747` | `0.733` | `0.493` | `4.85` |
| Trusted 10-turn rerun | `003b7c4d` | `0.341` | `0.240` | `0.347` | `0.973` | `0.333` | `0.333` | `0.240` | `9.28` |
| Old budget, high concurrency | `21159988` | `0.180` | `0.160` | `0.213` | `0.320` | `0.173` | `0.173` | `0.160` | `1.25` |

The high-concurrency control did remove the trusted-image import failure:
`ModuleNotFoundError` appeared in `0/75` rollouts, matching the old reference
and unlike the trusted-image rerun. However, it introduced a different failure
mode. `60/75` rollouts ended with `timeout_reached`, and `50/75` had zero tool
turns. The old reference had no timeout stops and no zero-turn rollouts.

This means the latest control does not falsify the old `0.618` result. It shows
that high requested concurrency overloads or starves the RLM rollout path,
causing many rollouts to time out before useful work starts. For a clean
apples-to-apples rerun of the old reference, use the same old budget with the
old/lower effective concurrency rather than `--max-concurrent 10`.

## GPT-5.5 Rerun: 10 Turns, 3 Checks, Non-Trusted Image, Concurrency 5

GPT-5.5 was rerun again with the larger RLM budget but without the trusted
sandbox image path that caused the dependency/interpreter mismatch. This run
used `25` examples, `3` rollouts per example, `max_turns=10`, `max_checks=3`,
`max_tokens=8192`, `trust_sandbox_image=False`, and `--max-concurrent 5`.

| Run | Local eval id | Mean reward | pass@1 | pass@2 | Format | Composition | Bond lengths | Formation energy | Avg turns | has_error |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Trusted 10-turn rerun | `003b7c4d` | `0.341` | `0.240` | `0.347` | `0.973` | `0.333` | `0.333` | `0.240` | `9.28` | `0.040` |
| Non-trusted, concurrency 5 | `34ff5575` | `0.516` | `0.440` | `0.520` | `0.960` | `0.520` | `0.520` | `0.440` | `7.64` | `0.107` |

This run confirms the main diagnosis from the rollouts: the large drop in the
trusted-image run was not an intrinsic effect of giving GPT-5.5 more turns and
checker calls. With `trust_sandbox_image=False`, `ModuleNotFoundError` occurred
in `0/75` rollouts, and the run recovered most of the gap to the original
5-turn baseline.

The final reward distribution was:

| Reward | Count |
|---:|---:|
| `1.0` | `33` |
| `0.4` | `6` |
| `0.1` | `33` |
| `0.0` | `3` |

The run still did not fully match the old `0.618` baseline. The remaining gap
looks behavioral and operational rather than dependency-related:

- `33/75` rollouts were full-credit, compared with `37/75` in the old
  5-turn/1-check reference.
- `37/75` rollouts reached `max_turns_reached`, showing that several samples
  still consumed the larger turn budget without converging to a passing CIF.
- `8/75` rollouts ended with `has_error`; these were HTTP 408 sandbox
  polling/upload errors, not Python import failures.
- `3/75` rollouts had zero turns, all from the same HTTP 408-style error path.
- Low-scoring completed rollouts were mostly format-only (`0.1`) or
  format+composition/bond without formation energy (`0.4`).

The practical conclusion is that the non-trusted image setting should be used
for current RLM evals until the trusted image launches the same dependency-rich
Python interpreter. `--max-concurrent 5` is much healthier than `10`, but still
not perfectly clean: the tail of this run had sandbox API 408s. If the goal is
an apples-to-apples replacement for the old reference, rerun with non-trusted
image, `max_turns=5`, `max_checks=1`, and a conservative concurrency of `3` to
`5`.

## GPT-5.5 Rerun: 15 Turns, 5 Checks, Non-Trusted Image, Concurrency 5

GPT-5.5 was then rerun with a larger per-rollout budget:
`max_turns=15`, `max_checks=5`, `max_tokens=8192`,
`trust_sandbox_image=False`, and `--max-concurrent 5`. The local eval id is
`60cf0943`.

| Run | Local eval id | Mean reward | pass@1 | pass@2 | Format | Composition | Bond lengths | Formation energy | Avg turns | has_error |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Trusted 10-turn rerun | `003b7c4d` | `0.341` | `0.240` | `0.347` | `0.973` | `0.333` | `0.333` | `0.240` | `9.28` | `0.040` |
| Non-trusted, 10 turns, 3 checks | `34ff5575` | `0.516` | `0.440` | `0.520` | `0.960` | `0.520` | `0.520` | `0.440` | `7.64` | `0.107` |
| Non-trusted, 15 turns, 5 checks | `60cf0943` | `0.783` | `0.733` | `0.760` | `0.907` | `0.840` | `0.840` | `0.733` | `8.95` | `0.093` |

This is the strongest RLM result so far. It improved mean reward by `+0.267`
over the 10-turn/3-check non-trusted run and by `+0.165` over the original
5-turn/1-check RLM reference. Full-credit rollouts increased from `33/75` in
the 10-turn/3-check non-trusted run to `55/75`.

The saved reward distribution was:

| Reward | Count |
|---:|---:|
| `1.0` | `55` |
| `0.4` | `8` |
| `0.1` | `5` |
| `0.0` | `7` |

The zero-reward cases in this run were not chemistry failures. All `7/75`
zero-reward rollouts had `SandboxNotReadyError` and zero turns. The non-perfect
completed rollouts were concentrated in a few examples:

- Example groups `13`, `16`, and `22` produced `0.4` partial-credit rollouts,
  usually satisfying format/composition/bond checks but missing formation
  energy.
- Example group `14` produced three `0.1` format-only rollouts and ended by
  timeout, so it remains a hard model-behavior case under this budget.
- Example groups `9` and `25` each had one `0.1` rollout while the sibling
  rollouts reached full credit.

The practical conclusion changed: increasing from `10/3` to `15/5` materially
helps GPT-5.5 in the RLM environment when the dependency-clean non-trusted image
path is used. The remaining limitation is now a mix of a few hard chemistry
examples and sandbox creation reliability at concurrency `5`, not the earlier
`ModuleNotFoundError` issue.

## Cross-Model Non-Trusted Budget Sweep

The non-trusted-image sweep was repeated across `openai/gpt-5.4-mini`,
`google/gemini-2.5-flash`, and `anthropic/claude-haiku-4.5` using the same
25x3 dataset slice, `max_concurrent=5`, `max_tokens=8192`, and the same four
rubrics. These runs make the budget effect clearer than the earlier
trusted-image diagnostics.

| Model | 10/3 local eval id | 10/3 mean | 10/3 pass@1 | 10/3 pass@2 | 15/5 local eval id | 15/5 mean | 15/5 pass@1 | 15/5 pass@2 | Delta mean |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|
| `openai/gpt-5.5` | `34ff5575` | `0.516` | `0.440` | `0.520` | `60cf0943` | `0.783` | `0.733` | `0.760` | `+0.267` |
| `openai/gpt-5.4-mini` | `0d02feeb` | `0.376` | `0.080` | `0.120` | `5dd698c6` | `0.408` | `0.107` | `0.173` | `+0.032` |
| `google/gemini-2.5-flash` | `1476c619` | `0.297` | `0.160` | `0.280` | `65760884` | `0.259` | `0.107` | `0.173` | `-0.038` |
| `anthropic/claude-haiku-4.5` | `01cf1c10` | `0.303` | `0.133` | `0.200` | `786ad9af` | `0.454` | `0.267` | `0.413` | `+0.151` |

The rubric-level view shows why mean reward and pass rates diverge across
models:

| Model | Budget | Format | Composition | Bond lengths | Formation energy | Avg turns | Errors |
|---|---|---:|---:|---:|---:|---:|---:|
| `openai/gpt-5.5` | `10/3` | `0.960` | `0.520` | `0.520` | `0.440` | `7.64` | `0.107` |
| `openai/gpt-5.5` | `15/5` | `0.907` | `0.840` | `0.840` | `0.733` | `8.95` | `0.093` |
| `openai/gpt-5.4-mini` | `10/3` | `1.000` | `0.973` | `0.547` | `0.080` | `7.64` | `0.000` |
| `openai/gpt-5.4-mini` | `15/5` | `1.000` | `1.000` | `0.627` | `0.107` | `9.28` | `0.000` |
| `google/gemini-2.5-flash` | `10/3` | `0.867` | `0.387` | `0.373` | `0.160` | `3.55` | `0.173` |
| `google/gemini-2.5-flash` | `15/5` | `0.947` | `0.347` | `0.320` | `0.107` | `4.07` | `0.040` |
| `anthropic/claude-haiku-4.5` | `10/3` | `0.933` | `0.467` | `0.400` | `0.133` | `8.89` | `0.067` |
| `anthropic/claude-haiku-4.5` | `15/5` | `1.000` | `0.680` | `0.613` | `0.267` | `13.59` | `0.000` |

Reward distributions for the newly completed non-GPT-5.5 runs were:

| Model | Budget | `0.0` | `0.1` | `0.25` | `0.4` | `0.85`/`0.9` | `1.0` |
|---|---|---:|---:|---:|---:|---:|---:|
| `openai/gpt-5.4-mini` | `10/3` | `0` | `2` | `31` | `36` | `1` | `5` |
| `openai/gpt-5.4-mini` | `15/5` | `0` | `0` | `28` | `39` | `0` | `8` |
| `google/gemini-2.5-flash` | `15/5` | `4` | `43` | `6` | `14` | `0` | `8` |
| `anthropic/claude-haiku-4.5` | `15/5` | `0` | `24` | `5` | `26` | `0` | `20` |

The sweep changes the model ranking under the larger RLM budget:

- GPT-5.5 is the only model that clearly converts extra checker and turn
  budget into high full-credit yield. Composition, bond-length, and formation
  energy rewards all move together from the `0.44` to `0.52` range into the
  `0.73` to `0.84` range.
- GPT-5.4-mini is reliable and syntactically clean, but most extra budget goes
  into partial credit. Formation energy improves only from `0.080` to `0.107`,
  so pass rates stay low despite perfect format and composition metrics.
- Claude benefits from the larger budget, improving mean reward from `0.303`
  to `0.454` and pass@2 from `0.200` to `0.413`. It is expensive in turns:
  average turns reached `13.59`, and the terminal summary showed `70.7%` of
  rollouts still hitting `max_turns_reached`.
- Gemini regressed from `0.297` to `0.259` when moving from `10/3` to `15/5`.
  The larger budget did not translate into better composition or energy
  validity; composition fell from `0.387` to `0.347`, formation energy fell
  from `0.160` to `0.107`, and the run still had a small model/agent error
  tail from empty responses.

There was no evidence in these new non-trusted runs of the earlier
`ModuleNotFoundError` or image dependency mismatch. `openai/gpt-5.4-mini` and
Claude had `0` average error in the saved metadata. Gemini had `0.040` average
error, attributed in the run summary to `EmptyModelResponseError` and
`AgentError`, not environment imports.

Automatic upload was skipped for these evals because the local
`crystal-relaxation-rlm` environment is ahead of the Hub copy. The saved local
metadata and `results.jsonl` files are therefore the authoritative artifacts
for this comparison.

## Infrastructure Notes

The RLM `gpt-5-mini` scaling runs with `max_turns=1` were intentionally
constrained smoke and setup-timing evals. Their near-zero rewards are expected:
the model usually spent the only turn reading the input CIF and did not have
enough turns to write `/task/final.cif`.

Those scaling runs are still useful infrastructure evidence:

| Eval | Local eval id | Avg reward | Infra failures |
|---|---|---:|---:|
| 1x1 | `outputs/evals/crystal_relaxation_rlm--openai--gpt-5-mini/3ac06a55/` | `0.0000` | `0` |
| 5x3 | `outputs/evals/crystal_relaxation_rlm--openai--gpt-5-mini/d2862d4f/` | `0.0067` | `0` |
| 10x3 | `outputs/evals/crystal_relaxation_rlm--openai--gpt-5-mini/1f33c926/` | `0.0000` | `0` |

The trusted-image path removed runtime dependency checks as a meaningful
bottleneck. Remaining setup cost is dominated by sandbox post-setup and agent
installation, with task/harness upload as a secondary cost.

## Reproduction Command

The full reruns used this command shape, changing only `--model`:

```bash
prime eval run crystal-relaxation-rlm \
  --env-dir-path .. \
  --model openai/gpt-5.5 \
  --num-examples 25 \
  --rollouts-per-example 3 \
  --max-concurrent 3 \
  --max-tokens 8192 \
  --env-args '{"dataset_path":"4everStudent/crystal-relaxation-mp20-with-clusters","max_examples":25,"max_turns":10,"max_checks":3,"trust_sandbox_image":true,"enabled_rubrics":["format","composition","bond_lengths","formation_energy"]}' \
  --save-results \
  --abbreviated-summary
```

The tested models were `openai/gpt-5.5`, `openai/gpt-5.4-mini`,
`google/gemini-2.5-flash`, and `anthropic/claude-haiku-4.5`.

The GPT-5.5 non-trusted concurrency-5 rerun used:

```bash
prime eval run crystal-relaxation-rlm \
  --env-dir-path .. \
  --model openai/gpt-5.5 \
  --num-examples 25 \
  --rollouts-per-example 3 \
  --max-concurrent 5 \
  --max-tokens 8192 \
  --env-args '{"dataset_path":"4everStudent/crystal-relaxation-mp20-with-clusters","max_examples":25,"max_turns":10,"max_checks":3,"trust_sandbox_image":false,"enabled_rubrics":["format","composition","bond_lengths","formation_energy"]}' \
  --save-results \
  --abbreviated-summary
```

The GPT-5.5 15-turn/5-check non-trusted concurrency-5 rerun used:

```bash
prime eval run crystal-relaxation-rlm \
  --env-dir-path .. \
  --model openai/gpt-5.5 \
  --num-examples 25 \
  --rollouts-per-example 3 \
  --max-concurrent 5 \
  --max-tokens 8192 \
  --env-args '{"dataset_path":"4everStudent/crystal-relaxation-mp20-with-clusters","max_examples":25,"max_turns":15,"max_checks":5,"trust_sandbox_image":false,"enabled_rubrics":["format","composition","bond_lengths","formation_energy"]}' \
  --save-results \
  --abbreviated-summary
```
