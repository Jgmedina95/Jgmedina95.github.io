# TL;DR Options for `blog_post.html`

## Option 1

This project trains language models with reinforcement learning to generate relaxed crystal structures directly as CIF files. A fine-tuned CHGNet model provides formation-energy and force signals, while separate rubric and heuristic checks handle validity, composition, and bond geometry, making crystal generation a verifiable RL task. The main result is that model scale changes not just performance, but reasoning style: the larger model develops richer crystallographic reasoning, while the smaller model compresses its reasoning and plateaus on the hardest physics objective.

## Option 2

I frame crystal relaxation as a verifiable reinforcement-learning problem where an LLM proposes a crystal structure, CHGNet evaluates formation energy and forces, and separate checks enforce validity, composition, and heuristic bond constraints. Across training runs, the 30B model becomes better at formatting and conservative edits, but the larger 230B model is the one that learns deeper structural reasoning and reaches substantially better formation-energy performance. The broader lesson is that, for this task, scale determines the kind of reasoning RL can induce.

## Option 3

This post shows how I built an RL environment for crystal generation in which models output CIF structures and receive reward from a verifier that combines a fine-tuned CHGNet model for formation energy and forces with separate format, composition, and bond-length checks. The reward design combines format, composition, forces, bond lengths, and formation energy, while the training analysis compares how 30B and 230B models behave under the same setup. The key takeaway is that larger models do not simply do better; they learn a qualitatively different, more scientifically grounded reasoning strategy.

## Option 4

I use GRPO to train language models to propose relaxed crystal structures in one shot, with a verifier that combines CHGNet formation-energy and force estimates with separate validity, composition, and heuristic bond checks. Comparing 30B and 230B runs reveals a clear capacity gap: the smaller model learns compressed procedural heuristics, while the larger model develops longer, more domain-aware reasoning and achieves much stronger stability rewards. In this setting, reward design matters, but model scale is what unlocks genuinely useful scientific reasoning.