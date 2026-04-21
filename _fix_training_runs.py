path = '/Users/jorgemedina/PersonalWebsite/Jgmedina95.github.io/GRPO_For_Materials/blog_post.html'

with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Find the training runs section by unique markers
section_start_marker = '<section class="prose prose-indigo lg:prose-xl dark:prose-invert mt-16">'
footer_marker = '<footer class="mt-16 pt-8 border-t border-gray-200 text-center">'

idx_start = content.find(section_start_marker)
idx_end = content.find(footer_marker)

if idx_start == -1 or idx_end == -1:
    print(f"Markers not found: start={idx_start}, end={idx_end}")
    exit(1)

old_section = content[idx_start:idx_end]
print(f"Old section length: {len(old_section)} chars")

new_section = '''        <section class="prose prose-indigo lg:prose-xl dark:prose-invert mt-16">
            <header class="mb-10 border-b border-gray-200 dark:border-gray-700 pb-6">
                <h2 class="text-3xl font-bold text-gray-900 dark:text-white mb-2">Training Runs: How an LLM \u201cLearns\u201d Physics</h2>
                <div class="flex items-center text-sm text-gray-500 dark:text-gray-400 italic">
                    <span>Posted on February 17, 2026</span>
                    <span class="mx-2">\u2022</span>
                    <span>Tagged:</span>
                    <span class="bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 px-2 py-0.5 rounded text-xs ml-2">RL</span>
                    <span class="bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 px-2 py-0.5 rounded text-xs ml-2">Materials Science</span>
                </div>
            </header>

            <p class="text-lg text-gray-700 dark:text-gray-300 mb-6">
                We ran the crystal relaxation environment on two model scales: a <strong>30B-parameter</strong> model (87 rollout samples, steps 0\u2013100, 11 checkpoints) and a <strong>230B-parameter</strong> model (63 samples, steps 0\u2013070, 8 checkpoints). Both runs use the same reward function, the same filtered MP-20 dataset from HuggingFace, and the same GRPO training loop. What we found is that scale doesn\u2019t just change <em>how well</em> a model learns\u2014it determines <em>what kind of reasoning strategy</em> RL selects for entirely.
            </p>

            <h3 class="text-2xl font-bold text-indigo-700 dark:text-indigo-400 mt-10 mb-4">1. Reward Trajectories</h3>

            <p class="mb-4">Both models improve over training, but the trajectories look completely different:</p>

            <div class="overflow-x-auto mb-6">
                <table class="min-w-full text-sm border rounded-lg overflow-hidden">
                    <thead class="bg-indigo-50 dark:bg-indigo-900/40 text-gray-700 dark:text-gray-200 font-semibold">
                        <tr>
                            <th class="px-4 py-2 text-left">Step</th>
                            <th class="px-4 py-2 text-center" colspan="2">230B Model</th>
                            <th class="px-4 py-2 text-center" colspan="2">30B Model</th>
                        </tr>
                        <tr class="text-xs text-gray-500 dark:text-gray-400">
                            <th class="px-4 py-1"></th>
                            <th class="px-4 py-1 text-center">Mean Reward</th>
                            <th class="px-4 py-1 text-center">Std</th>
                            <th class="px-4 py-1 text-center">Mean Reward</th>
                            <th class="px-4 py-1 text-center">Std</th>
                        </tr>
                    </thead>
                    <tbody class="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-700 text-gray-700 dark:text-gray-300">
                        <tr><td class="px-4 py-2">0</td><td class="px-4 py-2 text-center">0.706</td><td class="px-4 py-2 text-center">0.364</td><td class="px-4 py-2 text-center">0.406</td><td class="px-4 py-2 text-center">0.202</td></tr>
                        <tr class="bg-gray-50 dark:bg-gray-800"><td class="px-4 py-2">10</td><td class="px-4 py-2 text-center">0.705</td><td class="px-4 py-2 text-center">0.145</td><td class="px-4 py-2 text-center">0.710</td><td class="px-4 py-2 text-center">0.235</td></tr>
                        <tr><td class="px-4 py-2">20</td><td class="px-4 py-2 text-center">0.880</td><td class="px-4 py-2 text-center">0.164</td><td class="px-4 py-2 text-center">0.703</td><td class="px-4 py-2 text-center">0.214</td></tr>
                        <tr class="bg-gray-50 dark:bg-gray-800"><td class="px-4 py-2">30</td><td class="px-4 py-2 text-center">0.722</td><td class="px-4 py-2 text-center">0.270</td><td class="px-4 py-2 text-center">0.498</td><td class="px-4 py-2 text-center">0.272</td></tr>
                        <tr><td class="px-4 py-2">40</td><td class="px-4 py-2 text-center">0.502</td><td class="px-4 py-2 text-center">0.308</td><td class="px-4 py-2 text-center">0.727</td><td class="px-4 py-2 text-center">0.223</td></tr>
                        <tr class="bg-gray-50 dark:bg-gray-800"><td class="px-4 py-2">50</td><td class="px-4 py-2 text-center">0.719</td><td class="px-4 py-2 text-center">0.186</td><td class="px-4 py-2 text-center">0.696</td><td class="px-4 py-2 text-center">0.276</td></tr>
                        <tr><td class="px-4 py-2">60</td><td class="px-4 py-2 text-center font-bold text-indigo-700 dark:text-indigo-400">0.954</td><td class="px-4 py-2 text-center font-bold text-indigo-700 dark:text-indigo-400">0.116</td><td class="px-4 py-2 text-center">0.589</td><td class="px-4 py-2 text-center">0.275</td></tr>
                        <tr class="bg-gray-50 dark:bg-gray-800"><td class="px-4 py-2">70</td><td class="px-4 py-2 text-center">0.928</td><td class="px-4 py-2 text-center">0.144</td><td class="px-4 py-2 text-center">0.529</td><td class="px-4 py-2 text-center">0.267</td></tr>
                        <tr><td class="px-4 py-2">80</td><td class="px-4 py-2 text-center text-gray-400">\u2014</td><td class="px-4 py-2 text-center text-gray-400">\u2014</td><td class="px-4 py-2 text-center font-bold text-purple-700 dark:text-purple-400">0.766</td><td class="px-4 py-2 text-center font-bold text-purple-700 dark:text-purple-400">0.277</td></tr>
                        <tr class="bg-gray-50 dark:bg-gray-800"><td class="px-4 py-2">90</td><td class="px-4 py-2 text-center text-gray-400">\u2014</td><td class="px-4 py-2 text-center text-gray-400">\u2014</td><td class="px-4 py-2 text-center">0.661</td><td class="px-4 py-2 text-center">0.327</td></tr>
                        <tr><td class="px-4 py-2">100</td><td class="px-4 py-2 text-center text-gray-400">\u2014</td><td class="px-4 py-2 text-center text-gray-400">\u2014</td><td class="px-4 py-2 text-center text-red-600 dark:text-red-400">0.520</td><td class="px-4 py-2 text-center text-red-600 dark:text-red-400">0.258</td></tr>
                    </tbody>
                </table>
            </div>

            <p class="mb-6">The 230B model follows a <strong>non-monotonic but convergent</strong> trajectory: it dips at step 40 (mean 0.502) before recovering strongly to peak at step 60 (0.954) and sustaining near-peak performance at step 70 (0.928). Critically, its variance <em>contracts</em>\u2014std drops from 0.364 at step 0 to 0.116 at step 60, meaning the model becomes more consistent as it learns. By step 60, even its worst sample (reward 0.692) would have been above-average at step 0.</p>

            <p class="mb-6">The 30B model is more volatile. It peaks at step 80 (0.766) but <strong>regresses to 0.520 by step 100</strong>, with variance that never contracts below 0.20. Its best checkpoint is still 20% below the 230B model\u2019s peak. The single most dramatic data point: at step 30, the 30B model produces the only <strong>zero-reward sample</strong> in either dataset (SnF\u2083, where the model correctly identifies a symmetry mismatch but fails to produce any valid output).</p>

            <h3 class="text-2xl font-bold text-indigo-700 dark:text-indigo-400 mt-10 mb-4">2. Scale Determines Reasoning Strategy</h3>

            <p class="mb-4">The most surprising finding comes from tracking the length and content of the models\u2019 chain-of-thought reasoning blocks:</p>

            <div class="overflow-x-auto mb-6">
                <table class="min-w-full text-sm border rounded-lg overflow-hidden">
                    <thead class="bg-gray-50 dark:bg-gray-800 font-semibold text-gray-700 dark:text-gray-200">
                        <tr>
                            <th class="px-4 py-2 text-left">Step</th>
                            <th class="px-4 py-2 text-center">230B reasoning (chars)</th>
                            <th class="px-4 py-2 text-center">30B reasoning (chars)</th>
                        </tr>
                    </thead>
                    <tbody class="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-700 text-gray-700 dark:text-gray-300">
                        <tr><td class="px-4 py-2">0</td><td class="px-4 py-2 text-center">3,966</td><td class="px-4 py-2 text-center">1,532</td></tr>
                        <tr class="bg-gray-50 dark:bg-gray-800"><td class="px-4 py-2">20</td><td class="px-4 py-2 text-center">4,440</td><td class="px-4 py-2 text-center">1,502</td></tr>
                        <tr><td class="px-4 py-2">40</td><td class="px-4 py-2 text-center">5,969</td><td class="px-4 py-2 text-center">1,270</td></tr>
                        <tr class="bg-gray-50 dark:bg-gray-800"><td class="px-4 py-2">60</td><td class="px-4 py-2 text-center">6,640</td><td class="px-4 py-2 text-center">924</td></tr>
                        <tr><td class="px-4 py-2">70</td><td class="px-4 py-2 text-center font-bold text-indigo-700 dark:text-indigo-300">7,320 (+85%)</td><td class="px-4 py-2 text-center">810</td></tr>
                        <tr class="bg-gray-50 dark:bg-gray-800"><td class="px-4 py-2">80</td><td class="px-4 py-2 text-center text-gray-400">\u2014</td><td class="px-4 py-2 text-center">654</td></tr>
                        <tr><td class="px-4 py-2">100</td><td class="px-4 py-2 text-center text-gray-400">\u2014</td><td class="px-4 py-2 text-center font-bold text-purple-700 dark:text-purple-300">574 (\u221263%)</td></tr>
                    </tbody>
                </table>
            </div>

            <p class="mb-6">The 230B model\u2019s reasoning <strong>grows by 85%</strong> over training\u2014from 3,966 to 7,320 characters by step 70. The 30B model\u2019s reasoning <strong>shrinks by 63%</strong>\u2014from 1,532 to 574 characters. These are not just different amounts; they are opposite directions. At step 80, one 30B sample produced a reasoning block of <strong>3 characters</strong>\u2014essentially skipping thinking entirely and going straight to CIF output. By step 100, the 30B\u2019s <em>maximum</em> reasoning length (914 chars) is shorter than its step-0 <em>minimum</em> (1,288 chars).</p>

            <div class="bg-indigo-50 dark:bg-indigo-900/20 border-l-4 border-indigo-500 dark:border-indigo-400 p-6 my-8 rounded-r-lg">
                <h4 class="text-indigo-800 dark:text-indigo-200 font-bold mb-3">System 1 vs. System 2 Reasoning</h4>
                <p class="text-indigo-900 dark:text-indigo-100 mb-3">
                    This maps cleanly onto Kahneman\u2019s dual-process framework. The <strong>230B model develops System 2 reasoning</strong>: deliberative, step-by-step crystallographic analysis that deepens over training. By step 30, it correctly identifies hexagonal cells from \u03b3\u2009=\u2009120\u00b0; by step 60\u201370, it references Wyckoff positions, prototype structures, and Pnma space groups it was never explicitly trained on\u2014and explicitly acknowledges it cannot run real DFT while still producing near-perfect structures.
                </p>
                <p class="text-indigo-900 dark:text-indigo-100">
                    The <strong>30B model converges on System 1</strong>: compressed, procedural preambles (\u201crelaxation will be performed using DFT\u2014output below\u201d) that discard domain vocabulary in favor of getting directly to the CIF. RL selects for this compression because the smaller model lacks the capacity to maintain both elaborate reasoning <em>and</em> correct structural output simultaneously. By step 90\u2013100, the reasoning reads as a <em>formality</em> rather than a problem-solving step\u2014the model even inserts the parenthetical \u201c(simulated here for brevity)\u201d.
                </p>
            </div>

            <p class="mb-6">The keyword data confirms this divergence. In the 230B model, \u201csymmetry\u201d mentions grow from 6.4 per sample at step 0 to <strong>11.3 at step 60</strong>, and Wyckoff positions emerge at step 40+. In the 30B model, \u201csymmetry\u201d starts at 0.6 and reaches <strong>0.0 by step 100</strong>. \u201cBond\u201d drops from 2.0 to 0.0 by step 60. \u201cCoordination\u201d and \u201cWyckoff\u201d never appear at any step.</p>

            <h3 class="text-2xl font-bold text-indigo-700 dark:text-indigo-400 mt-10 mb-4">3. The \u201cBond\u201d Paradox</h3>

            <div class="bg-purple-50 dark:bg-purple-900/20 border-l-4 border-purple-500 dark:border-purple-400 p-6 my-6 rounded-r-lg">
                <p class="text-purple-900 dark:text-purple-100 mb-3">
                    The 30B model\u2019s \u201cbond\u201d keyword count drops from <strong>2.0 per sample at step 0 to 0.0 by step 60</strong>\u2014and stays at zero through step 100. On the surface, this looks like the model forgetting about bond lengths.
                </p>
                <p class="text-purple-900 dark:text-purple-100">
                    In reality, the <strong>bond length reward improves from 0.574 to 0.809</strong> over the same period (peaking at 0.818 at step 90). The model\u2019s bond quality gets better while it stops talking about bonds. This is the compression heuristic in action: the concept has been internalized into the structural output itself, removed from the scratchpad. The 30B model learns to <em>do</em> without needing to <em>say</em>. The 230B model does the opposite: bond mentions also decline after step 50, but from a higher baseline and after the model has already peak-loaded its symmetry vocabulary.
                </p>
            </div>

            <h3 class="text-2xl font-bold text-indigo-700 dark:text-indigo-400 mt-10 mb-4">4. Conservative Perturbations\u2014Two Very Different Kinds</h3>

            <p class="mb-4">Both models learn to make smaller lattice adjustments over time through step 60. But the degree, pattern, and outcome differ sharply:</p>

            <div class="overflow-x-auto mb-6">
                <table class="min-w-full text-sm border rounded-lg overflow-hidden">
                    <thead class="bg-gray-50 dark:bg-gray-800 font-semibold text-gray-700 dark:text-gray-200">
                        <tr>
                            <th class="px-4 py-2 text-left">Model</th>
                            <th class="px-4 py-2 text-center">\u0394a at Step 0</th>
                            <th class="px-4 py-2 text-center">Min \u0394a</th>
                            <th class="px-4 py-2 text-center">Reduction</th>
                            <th class="px-4 py-2 text-center">FE score at reward peak</th>
                        </tr>
                    </thead>
                    <tbody class="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-700 text-gray-700 dark:text-gray-300">
                        <tr>
                            <td class="px-4 py-2 font-medium">230B</td>
                            <td class="px-4 py-2 text-center">0.053 \u00c5</td>
                            <td class="px-4 py-2 text-center">0.017 \u00c5 (step 60)</td>
                            <td class="px-4 py-2 text-center">\u221268%</td>
                            <td class="px-4 py-2 text-center font-bold text-indigo-700 dark:text-indigo-300">0.957 (step 60)</td>
                        </tr>
                        <tr class="bg-gray-50 dark:bg-gray-800">
                            <td class="px-4 py-2 font-medium">30B</td>
                            <td class="px-4 py-2 text-center">0.025 \u00c5</td>
                            <td class="px-4 py-2 text-center">0.002 \u00c5 (step 90)</td>
                            <td class="px-4 py-2 text-center">\u221291%</td>
                            <td class="px-4 py-2 text-center text-orange-600 dark:text-orange-400">0.658 (step 80)</td>
                        </tr>
                    </tbody>
                </table>
            </div>

            <p class="mb-6">The 230B model\u2019s conservatism is <em>healthy</em>: it mirrors real DFT relaxation behavior, correlates with excellent formation energy scores, and notably <strong>rebounds at step 70</strong> (\u0394a = 0.084 \u00c5)\u2014the model becomes more forceful once it has the structural reasoning to support larger corrections, and bond quality peaks at 0.938 that same step. The 30B model\u2019s conservatism is <em>pathological</em>: modifying the lattice by 0.002 \u00c5 is essentially returning the input unchanged. It avoids catastrophic bond-length violations but cannot correct misplaced lattice parameters\u2014hence formation energies that remain 10\u201320\u00d7 above the 0.1 eV/atom threshold throughout all 100 steps.</p>

            <h3 class="text-2xl font-bold text-indigo-700 dark:text-indigo-400 mt-10 mb-4">5. Formation Energy: The Capacity Bottleneck</h3>

            <p class="mb-6">The formation energy reward component is where the capacity gap is most visible. The 230B model reaches a formation energy score of <strong>0.957 at step 60</strong> and sustains 0.896 at step 70. The 30B model peaks at <strong>0.658 at step 80</strong> and falls back to 0.283 by step 100\u2014a ~45-percentage-point gap at peak, widening to 61 points by the end of each run.</p>

            <p class="mb-6">In absolute terms, the 30B model\u2019s generated structures have mean formation energies oscillating between 0.93 and 1.86 eV/atom across all steps\u2014<strong>10\u201320\u00d7 above the 0.1 eV/atom stability threshold</strong>. The structures pass composition and bond-length checks, but they are not genuinely relaxed. The model has learned to be a competent CIF formatter and a cautious structure perturber, but not a crystal-structure optimizer.</p>

            <h3 class="text-2xl font-bold text-red-700 dark:text-red-400 mt-10 mb-4">6. The Failure Mode: The \u201cRetry\u201d Illusion</h3>

            <p class="mb-6">
                In multi-turn rollouts, when the first attempt fails the energy threshold, the model gets a second turn with the validator\u2019s feedback. Analysis of 37 first-turn failures from the 30B run:
            </p>
            <ul class="list-disc list-inside mb-6 space-y-2 text-gray-700 dark:text-gray-300">
                <li><strong>Physical recoveries (energy, forces, bonds):</strong> 0. Zero cases of fixing bad physics.</li>
                <li><strong>Syntax recoveries (format corrections):</strong> 7. The model can fix a missing CIF line, but not a bad structure.</li>
            </ul>

            <p class="mb-6">
                The model treats scalar feedback (\u201cFormation energy is too high\u201d) as a signal to re-roll rather than to diagnose. It rephrases its reasoning and outputs a near-identical CIF with minor random perturbations. It has learned <strong>state recognition</strong> (what a valid crystal looks like) but not <strong>gradient estimation</strong> (which direction to move atoms to lower energy).
            </p>

            <p class="mb-6">
                There is one exception: <strong>prediction</strong> improves over turns. Even though the model cannot fix a bad structure, its ability to estimate formation energy from the feedback context improves in the second turn\u2014an \u201cActor-Critic\u201d disconnect where the Critic (prediction) improves while the Actor (structure generation) stalls.
            </p>

            <blockquote class="border-l-4 border-gray-300 dark:border-gray-600 pl-4 italic text-gray-600 dark:text-gray-400 my-8 py-2">
                \u201cThe model learns to measure the problem before it learns to fix it.\u201d
            </blockquote>

            <div class="bg-gray-50 dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 pt-6 mt-8">
                <p class="font-bold text-gray-900 dark:text-white mb-2">What this implies for the next run:</p>
                <p class="text-gray-700 dark:text-gray-300">
                    Scalar feedback (\u201cenergy too high\u201d) provides no directional gradient in atom-coordinate space. The path forward is <strong>vector feedback</strong>\u2014per-atom force vectors, specific bond-length violations, which Wyckoff site is strained\u2014so the policy has an actual direction to move. The 230B model shows that with sufficient capacity, RL can induce genuinely calibrated structural reasoning. Whether a 30B model can exploit richer feedback to close the gap, or whether the formation energy bottleneck is a hard capacity limit, is the central open question.
                </p>
            </div>

        </section>
'''

content = content[:idx_start] + new_section + content[idx_end:]

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Done. New section written.")
print(f"Total file length: {len(content)} chars")
