// NEXUS COGNITIVE ENGINE v13 — model-based RL research build (v12 + --auto: one configuration for every game)
// C++17, standard library only. No pretrained model. This is NOT AGI: it is an online
// model-based RL agent (learned world model + planner) built for falsifiable experiments.
//
// Build (recommended; the two extra flags let the compiler vectorise the dot products, ~2-3x faster):
//   g++ -std=c++17 -O3 -march=native -fassociative-math -fno-signed-zeros -fno-trapping-math nexus_cognitive_engine.cpp -o nexus
//   (plain  g++ -std=c++17 -O3 -march=native  also works and is bit-reproducible across code changes with -ffp-contract=off)
// Self-tests (98 checks, incl. finite-difference gradient checks, save/load, invariants, ASan/UBSan-clean):  ./nexus --selftest
// Best known setting, one flag:   ./nexus --preset stable --seed 1      (v7; ~30-60 min per run)
//   = 'best' + 500 extra episodes (1500 total) with the stabilisers below, tested on 60 held-out structures (4 episodes each).
// Previous setting (v6):          ./nexus --preset best --seed 1
// Existing game (v8):             ./nexus --preset mountaincar --seed 1     (MountainCar-v0, ~10 min per run, see 'Games' below)
//   = dense reward + learned value as planner tail, no hand prior, 1000 episodes on 50 generated task structures,
//     frozen final evaluation on 20 held-out structures (6 episodes each). Takes ~10-20 min per run.
//
// Stabilisers added in v7 (all optional flags; --preset stable switches them on together):
//   --lr-decay A B   learning rate falls linearly from 1x at episode A to 0.1x at episode B (preset: 1000 -> 1500);
//   --avg TAU        evaluation uses an exponential moving average of the weights (preset 1e-4, ~10k gradient updates);
//   --val-ids / --best-every / --best-eps / --best-from / --val-start
//                    best-snapshot selection on SEPARATE validation structures (preset: 40 structures x 6 episodes, every 100
//                    episodes from episode 1000; the starting snapshot is a candidate). Earlier builds validated on the test
//                    structures (only the start positions differed) - do not do that.
//   Measured (success on 60 held-out structures, 4 episodes each; paired seeds; chunked runs, 1000 episodes -> +500 with the stabilisers):
//     seeds 1-6 (used to choose the protocol):        77.8 -> 83.1   (+5.3, 95% CI +0.5..+9.2, 5 of 6 seeds better)
//     seeds 7-12 (blind confirmation, nothing tuned): 78.2 -> 81.5   (+3.3, 95% CI +1.0..+5.3, 5 of 6 seeds better)
//     all 12 seeds: 78.0 -> 82.3 (+4.3, CI +1.7..+6.7, 10 of 12). Chains (two keys) 64 -> 76, key-door 88 -> 91.
//     --preset stable as one run from scratch (seeds 13, 14, not used for any choice): 82.5% and 89.6%.
//   Not a cure for seed spread: sd between seeds 4.0 -> 4.3 (worst seed 72.1 -> 75.0). Training +500 episodes at a constant
//   learning rate gave no gain (77.8 -> 77.7, sd 3.4 -> 8.2); the schedule and the selection are what help. Re-running one
//   checkpoint with different random seeds gives 68-79%, i.e. ~5 points of run-to-run noise that no setting here removes.
//   Validation itself is noisy (~+-3 points at 40x6 episodes); the planner's own sampling noise is part of it.
//
// Games (v8; scenario ids >= 3000, any of --train-ids / --final-ids; the agent sees only the game's own observation and reward):
//   3000 MountainCar-v0 (Gym physics: force .001, gravity .0025, speed limit .07, flag at .5, 200 steps). Observation = position and
//     velocity only. Reward: -0.01 per step (a rescaled -1) and +4 on reaching the flag. The 9 arena actions are reduced to left / none /
//     right by their horizontal part. Reference policies: random 0%, 'push along the velocity' 100%.
//     Frozen agent, 50 episodes, learned from scratch in 600 episodes (6 seeds, one run each; configurations tried after looking at seed 1,
//     so seeds 4-6 are not blind for the last row; the official gymnasium package could not be installed here, the game is re-implemented):
//       horizon 8 (default), no novelty:             0%  (1 seed)
//       horizon 8 + --rnd 1.0:                       12% (1 seed)
//       --horizon 20 --gamma .99:                    98 / 0 / 66 / 50 / 2 / 0   mean 36
//       --horizon 20 --gamma .99 --rnd 1.0 (= --preset mountaincar):  98 / 100 / 56 / 98 / 100 / 100   mean 92
//       paired difference of the last two rows: +56 points, 95% CI +17..+91. Failing runs never reached the flag in training (no signal to learn from);
//       novelty (RND) fixes exactly that. In the arena RND gave nothing; with a sparse far goal it is the key ingredient.
//   3001 NumberChain: reach a target number (20..200) from 1 with +1 +2 +4 +8 x2 x3 -1 -4 /2 within 20 steps; the exact optimum comes from
//     breadth-first search (reference 100%, random 0%). A test of whether the agent can plan a short exact chain.
//     Frozen agent, 100 random targets, learned from scratch in 3000 episodes (default settings, horizon 8): 45 / 40 / 39% (seeds 1-3, mean 41%);
//     horizon 12 + gamma .97: 10% (1 seed). So the agent finds the target about 4 times in 10 within 20 steps, far from the exact optimum; it never
//     reached the reference. The model has to resolve differences of 1/128 in its input, which a small tanh MLP does poorly - an honest limit.
//     No dedicated preset: ./nexus --episodes 3000 --value --no-heuristic --no-label --train-ids 3001 --final-ids 3001 --final-eps 100
//
//
// v9 additions (all OFF by default; with every new flag off the agent is the v8 agent). Self-tests: 84/84.
//   --vq K            discrete symbols: scalar VQ codebooks per active observation dim, categorical next-symbol model, exactly
//                     linear-dependent dims are detected and recomputed. NumberChain (3001): 45/40% -> 96/95% (seeds 1-2, seed 3 pending).
//                     WORKS. (--tok-diag prints per-dim symbol accuracy.)
//   --hier NC         hierarchical planner: manager picks one of NC learned concepts (k-means abstract states) or a null option (flat CEM);
//                     option-level Q trained semi-MDP, CEM gets a potential-based progress reward. MEASURED NEGATIVE: MountainCar
//                     always-subgoal version 28% vs 86% baseline (-61, CI -100..+12); with null option 49% (-37, CI -78..+3, 2/5 seeds better).
//                     Kept for reproducibility; do not use it as is.
//   --curious BETA    intrinsic reward from a learned model-error map. With sparse reward on MountainCar (no RND): 100/100 vs 0/90
//                     (2 seeds, +55, CI +10..+100 - too few seeds to be sure). Blind (reward hidden, --blind) the flag was reached by
//                     chance in 10% of training episodes vs RND 35% vs none 0%: RND is the better explorer here.
//   --graph NC --graph-w W   persistent transition-graph memory: concept nodes, decaying edges, value iteration (detour around a blocked
//                     edge is unit-tested), pinned replay exemplars that survive floods (--rcap sets replay size). Forgetting benchmark
//                     on the arena did not finish in time - effect on real tasks is UNMEASURED, only the unit tests are.
//   --blind           the agent never sees reward/done (success still counted).
//   Not AGI: every item is a narrow mechanism, tested on a few small tasks with few seeds.
//
//
// v10 additions (OFF by default; every measurement below is paired seeds, small n, stated as it is):
//   --synth           symbolic rule synthesis: per (action, dimension) a program  ns = alpha*s + beta  with range/lattice guards is accepted only
//                     if it explains >= 99.5% of >= 30 samples. Accepted rules override the neural prediction, the rest stays neural.
//   --rules FILE      hypotheses (action dim alpha beta per line, '#' comments) written by a person or a language model; each one is
//                     checked against the agent's own experience and refuted ones are dropped (reported at the end of a run).
//                     Self-test: a true hypothesis is confirmed, a false one is refuted. NO language model runs inside this program.
//     Measured, NumberChain, 1500 episodes, --vq 256 --synth --rules rules_numberchain.txt, 100 frozen targets, seeds 1-4:  100 / 100 / 100 / 100 %
//     (earlier builds: plain 45 / 40 / 39 %, --vq 256 alone 96 / 95 %; BFS optimum = 100 %). Limit: it only helps where the dynamics ARE a short
//     per-dimension affine rule. MountainCar (cos term) and the arena (wall bounces, coupled dimensions) get no rules and are unchanged.
//   --bottleneck      with --hier: only concepts that successful episodes visit clearly more often than failed ones may be sub-goals.
//     Measured NEGATIVE: MountainCar 92 / 0 / 2 % (seeds 1-3) vs 100 / 100 / 68 % for RND alone. The hierarchy (v1, v2, bottleneck) did not
//     help on any task tested; do not use --hier as is.
//   Scale of structures (arena, 1000 episodes, 20 held-out structures x 6 episodes, seeds 1-2): 50 training structures 81.7 / 85.0 %,
//     200 training structures 86.7 / 82.5 %  -> no measurable gain at this budget (a thousand worlds is beyond 2 cores).
//   Forgetting (arena, replay 8000, learn structures A then B, success on A; seeds 1-2): ring buffer only 93/82 -> 63/58 (-26.7 points);
//     --graph 64 --graph-w 0.5 (pinned exemplars + value): 87/77 -> 93/63 (-3.3). Paired difference after B: +17.5, 95% CI +5..+30, n = 2 seeds
//     only (A after A was 93/82 vs 87/77, so the starting points differ too). Promising, not established.
//   Evaluation: nexus_eval_suite.sh NAME "flags" "seeds" runs the held-out arena / MountainCar / NumberChain benchmarks;
//     nexus_paired.py BASE VARIANT prints the paired difference with a bootstrap interval.
//   Still not AGI.
//
//
// v11 additions (OFF by default; the numbers marked MEASURED are in the notes at the end of this block):
//   --synth-lib       formula search over a library {s_k, s_k^2, s_k*s_m, sin(w s_k), cos(w s_k)} by sparse regression (orthogonal matching
//                     pursuit), sin and cos partners share one refined frequency, smallest formula that explains >= 99% of a HELD-OUT half is kept,
//                     clamped to the observed range. --synth-diag (with --eval-only --resume) prints every rule and every near miss.
//                     On MountainCar replay it recovers the position update x' = x + 0.0775 v (the printed formula carries small extra terms that
//                     fit noise) but NOT the velocity update (93% at best); on the arena it finds nothing at all (physics parameters are hidden and
//                     differ between structures, so no global rule exists). It does not help where it cannot find a rule.
//   --synth-report F  writes the dimensions no rule explains, with sample rows, for a person or a language model to read.
//   --rules F         now also takes formula lines:  f ACTION DIM = c + c1*s2 + c2*sin(w*s0) + c3*cos(w*s0) + c4*s1*s2 + c5*s3^2 + c6*s1
//                     Each is tested on ALL experience with that action and CONFIRMED (>= 99% within tolerance) or REFUTED, reported at the end.
//                     Closed loop done once by hand: the report for MountainCar -> formulas derived from the Gym equations by the proposer
//                     (rules_mountaincar.txt) -> 18/18 confirmed at 99-100% of 8000-15000 samples each. The proposer was me, not a program;
//                     no language model runs inside nexus.
//   Games (scenario ids): 3002 Pendulum swing-up (Gym dynamics, torque -2/0/+2, goal upright and slow; random 25%, reference controller 100%),
//                     3003 Switches (8 lights, a press flips the button and its neighbours; BFS optimum 100%, random 20%),
//                     3004 ModChain (reach a target modulo 97 with +1 +5 +17 x2 x3 -1 -7 x5 2x+1; BFS optimum 100%, random 15%).
//   nexus_parallel.sh runs many seeds on all cores (nexus is single-threaded; there is no GPU code).
//   nexus_eval_suite.sh / nexus_paired.py: held-out benchmark suite and paired comparison with bootstrap intervals.
//   Transfer between games: transfer.sh style runs (copy a trained checkpoint, --resume on another game) - see MEASURED notes.
//
//   MEASURED (v11; frozen test, seeds as listed, one run each):
//     MountainCar 600 episodes, --rnd 1.0:            RND alone 100 / 100 / 68 (seeds 1-3)
//       + --synth-lib (search for formulas by itself): 4 / 98 (seeds 1-2)  -> no gain, one seed collapsed. Do not use it on MountainCar.
//       + --rules rules_mountaincar.txt (formulas from the proposer, confirmed by the agent): 100 / 100 / 96 -> +9 on average, 3 seeds,
//         interval too wide to call it an improvement; at least it did not hurt.
//       + the same with the wall-reset clause (rules_mountaincar.txt as shipped, all 18 formulas confirmed on 100% of the stored samples):
//         100 / 100 / 100 -> +10.7 on average over the RND-only runs, but only seed 3 differs (68 -> 100). Three seeds: a hint, not a proof.
//     Switches (8 lights) 1500 episodes, 100 frozen starts, seed 1: plain 81%, --vq 256 100%, --vq 256 --synth-lib 100% -> the symbols do the work.
//       (The synthesis did find 33 value-changing rules = the toggle rules, but adding them did not raise 100%.) BFS optimum 100%.
//     Pendulum swing-up 800 episodes, 100 frozen starts, seed 1 (MountainCar flags): plain 100%, --synth-lib 94% (found no value-changing rule:
//       the observation holds cos/sin of the angle, so the update is not a short formula of the observation). Plain agent already solves it.
//     Transfer, one seed: NumberChain-trained --vq agent continued on ModChain reaches 48 / 66 / 96% after 500 / 1000 / 2000 episodes, a fresh
//       --vq agent 20 / 54 / 100%. Faster at the start, no difference at the end; one seed, noise is large - NOT established.
//     ModChain 2000 episodes, 100 targets, seed 1: plain 35%, --vq 256: 97%, --vq 256 --synth-lib: 99% (the symbols do the work; modular
//       wrap-around is not an affine rule, so synthesis found nothing useful there; BFS optimum 100%).
//     Transfer MountainCar -> Pendulum (one seed, both with the MountainCar flags): the final frozen test is 100% either way, but the learning
//       was SLOWER after transfer (training success averaged over the first 400 episodes 39.5% vs 67.5% for a fresh agent): negative transfer
//       early on, nothing later. So experience from one game did not help another here; the ModChain case above is the only hint of the opposite.
//     Forgetting chain (arena, replay 8000, four phases of 5 structures each, 300 episodes per phase, frozen test on all 20 structures, seeds 1-2):
//       after phase D the mean success over all 20 structures was ring buffer 70 / 75% (mean 72.5) and --graph 64 --graph-w 0.5: 80 / 75% (77.5);
//       structures of phase A: ring 80 / 90%, graph 100 / 90%. A +5 point hint over two seeds, 6 test episodes per structure: NOT established, and much
//       smaller than the 2-phase result above (-26.7 vs -3.3), so that result should not be read as a general property.
//
// v13 addition: --auto  (no per-game flags)
//   Equivalent to  --value --no-heuristic --no-label --synth-lib --horizon 20 --gamma 0.99  plus two decisions the agent takes itself from a probe of
//   3000 random moves on the training tasks (deterministic in the seed): all observation dimensions take <= 600 distinct values -> symbols (--vq 256);
//   the goal was never reached by chance -> novelty bonus (--rnd 1.0). The probe line is printed ("auto: ...").
//   MEASURED, one command per game, only ids differ, 1500 episodes (arena 1000), 100 frozen tests (arena 20 held-out structures x 6), seed 1:
//                    --auto     best hand-tuned configuration found earlier
//     MountainCar     47%        100 / 100 / 100 (rules file) , 100 / 100 / 68 (RND alone)
//     NumberChain     90%        100% (--vq + --synth + rules file)
//     Pendulum        85%        100% (MountainCar flags, rnd 1.0)
//     Switches       100%        100% (--vq 256)
//     ModChain        94%        97-99% (--vq 256)
//     Arena (held-out) 65%       82-85% (stable settings, horizon 8, gamma .94)
//   Mean over the six: ~80% against ~95%. So giving up the per-game tuning costs about 15 points, most on MountainCar and the arena. The fixed
//   horizon 20 / gamma .99 inside --auto is itself a choice made on MountainCar and is the likely reason for the arena result (the hand-tuned arena
//   runs used horizon 8 / gamma .94); making those two adapt is the obvious next step. One seed per game: differences of a few points are noise.
//
// v12 additions:
//   Rotation rules (with --synth-lib): a pair of observation dims on the unit circle (cos, sin of an angle) is advanced by a fitted angle increment;
//     the library gets a feature "the value dim k takes after this step by ITS OWN verified rule" (so chained physics such as angle += dt * new velocity
//     is expressible). Samples at a clamp limit are kept out of the least-squares fit (they biased it) but stay in the held-out test.
//   --synth-probe ID: collects random-policy transitions from scenario ID and prints what the synthesis finds, without training.
//   MEASURED on Pendulum (obs = cos, sin, w/8): from the agent's own replay it recovers  s2' = s2 + 0.09375 sin(theta) + 0.0375*u  and
//     angle' = angle + 0.4 * next(s2)  (the Gym equations: 15*dt/8 = 0.09375, dt*8 = 0.4), 100% of the held-out half.
//     Learning (800 episodes, MountainCar flags, 100 frozen starts: 100% for both): training success averaged over the first 400 episodes was
//       fresh agent 67.5% (1 seed) vs 63.0 / 61.8% with --synth-lib (2 seeds) - slower at the start (rules appear only after 5000 steps and the
//       first fits), then faster: success at episode 300 / 400 was 78 / 74% vs 88 / 86% and 90 / 96%. Mean over all 800 episodes 73.2% vs 77.8 / 78.5%.
//       A modest gain after the first 200 episodes, 2 seeds against 1: NOT established.
//   Earlier note about Pendulum (94%, no rule found) was produced before the clamp fix and no longer applies.
//
// What is in here (every item was measured; see the notes below):
//   world model: 3-member MLP ensemble (96x96 tanh), short-term velocity/previous-action context, optional multi-step
//     (unrolled) training (--unroll K), optional reward net on distance features (--rnet);
//   planner: CEM / MPC with warm start, learned double-Q tail (--value), macro-actions (--repeat), novelty (--rnd);
//   memory: hand-made coverage + goal memory (--cover, --goal-memory), fixed echo-state reservoir (--reservoir);
//   tasks: hidden-goal search, key-door, two-key chain, mirrored controls, drifting goal, wind; a procedural family of
//     task structures (scenario ids 100+index; ranges like 100-149 in --train-ids / --final-ids);
//   diagnostics: oracle reward/value, multi-step prediction error (--ms-diag), episode traces (--trace), best-snapshot
//     selection by held-out validation (--best-every), eval-only mode with other planner settings (--eval-only).
//
// Measured results (6 seeds unless noted; success on 20 held-out generated structures, greedy oracle = 100%, random ~12%):
//   training structures x episodes, 48 neurons:   2 x 320 = 59%   50 x 320 = 56%   2 x 1000 = 57%   50 x 1000 = 70%   50 x 2000 = 77%
//   width 96 (this build's default), 50 x 1000 (4 seeds): 80%.   192 neurons x 3000 episodes x 60 structures (2 seeds): 78% (plateau).
//   => diversity helps only together with enough episodes and capacity; each alone gives ~nothing.
//   Training without one structure type (two-key chains) still gives 24% on it (random ~1%) vs 63% when it was trained.
//   Negative / null results (kept as flags for reproducibility): RND novelty, domain randomisation, larger fixed reservoir,
//   multi-step training and reward net on key-door (seed spread too large to call), longer-horizon planning, best-snapshot selection.
// Known limits: seed-to-seed spread is large (sd ~6 points on the 20-structure mean; ~20 on small tasks); runs are
//   learning-speed lotteries, so always compare configurations on paired seeds.
// v14-v19 (latest; all numbers are single runs, seed 1, so rough):
//   v14  --auto also adapts to the reward: a step-reward std > .02 (shaped reward) -> horizon 8, gamma .94 and no novelty bonus. Synth rules that do not
//        beat the network on held samples are pruned. MEASURED: --auto on the arena (50 train / 20 held-out structures, 1000 episodes) 90.8%
//        (v13: 65%, hand-tuned about 83%). MountainCar --auto: training success about 90-96% (frozen 100-episode test still pending when this was written).
//   v15  cosmetic: misleading indentation, pi constants, memcmp comparison.
//   v16  held-out family: scenario ids 3100..3119 = 20 fixed puzzles (lights x4, modular chains x4, grids x2, sorting x2, hanoi x2, registers x2,
//        locks x2, 64-state graphs x2) with a BFS oracle. Built so that the agent and its author never tuned on them. Not yet evaluated at scale.
//   v17  speed: the MLP forward pass (97% of run time) uses an 8-lane dot product, about 3x faster; results equal older builds up to rounding.
//   v18  external environments: --ext "command" with scenario id 3200 (nexus_gym_bridge.py speaks the line protocol; needs gymnasium for real
//        Gym environments, `fake:CartPole` is built in for tests). MEASURED: CartPole (fake, Gym equations), 600 episodes: 0% frozen success
//        (agent did not solve it; see v19).
//   v19  failures end the Q target (Transition.term; before, the value target looked past a failed episode, which blurred the failure penalty);
//        the final summary prints the mean over all evaluated scenarios instead of the hand-made-seen-only 0.0%.
//   v20  --auto: reward density judged by the share of steps off the usual value (CartPole is not dense); world-model step scales (DSCALE) derived from
//        the probe where a step changes a dimension far less than the arena default (CartPole/MountainCar/Pendulum); periodic test runs on --final-ids.
//        MEASURED: CartPole model error 0.00002 vs 0.00275 for 'nothing changes' (was no better), but frozen success only 2% (1 of 50), mean 74 of 200
//        steps (random about 21): NOT SOLVED. MountainCar with these scales: training success 38% at episode 50, frozen test not finished.
//   v21  --rw-diag (with --eval-only --resume): does the reward head recognise failures and goals? Written to test the hypothesis that CartPole fails
//        because the head cannot see the sharp fall threshold; the check itself was NOT run before this file was packed.
//   v20  --auto: reward "dense" only if the reward differs step to step (CartPole is not); step-size scales (DSCALE) derived from the probe when one
//        step moves a dimension far less than the arena defaults (CartPole model error 0.021 -> 0.00002, 100x better than "nothing changes");
//        the periodic test runs on --final-ids instead of the three hand-made arena scenarios. MEASURED (seed 1, rough): CartPole (fake, 600 episodes)
//        frozen success 2% (1 of 50), mean 74 of 200 steps (random about 21): improved, but still unsolved. MountainCar --auto with scales: run interrupted.
//   v21  --rw-diag (with --eval-only --resume): reward-head check. MEASURED on CartPole: failure steps recognised 93%, so the head was NOT the problem.
//        The real cause was --auto's novelty bonus: with no goal found by chance it switched the bonus on, which pulls the agent toward the edges
//        where the pole falls. Now the bonus stays off when the probe sees failures (episodes ending badly). MEASURED (seed 1, 400 episodes, fake
//        CartPole): frozen success 100% (50/50 episodes, all 200 steps), up from 2%. Worlds without failures (MountainCar, arena, puzzles) decide as before.
//   v22  one self-contained file: nexus_gym_bridge.py and nexus_llm_hints.py are embedded; --write-bridge FILE / --write-llm-hints FILE write them out.
//   Not an AGI: a narrow research agent. Honest limits are listed in the sections above.
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <regex>
#include <sstream>
#include <map>
#include <unordered_map>
#include <string>
#include <vector>

namespace nx {

constexpr int OBS = 10, ACT = 9, CTX = 2, MEM = 32, CIN = OBS + CTX + MEM, IN = CIN + 2 * ACT, H1 = 96, H2 = 96, OUT = OBS + 1;
constexpr int RH = 32, RO = 16;  // RND hidden / output width
constexpr int QH = 64;           // Q-network hidden width
constexpr int ENSEMBLE = 3, REPLAY_CAP = 100000, MAXH = 32, MAXPOP = 128;
constexpr int BATCH = 32, WARMUP = 600, MAX_STEPS = 240, GAME_BASE = 3000, GAME_STEPS = 200, HELD_BASE = GAME_BASE + 100, HELD_N = 20, HELD_STEPS = 50;  // GAME_BASE+0: MountainCar-v0 (200 steps like the original TimeLimit)
constexpr float LR = .001f, LR_Q = .001f, LR_RND = .001f, REWARD_SCALE = 4.f, REWARD_LOSS_W = 2.f;
constexpr float EXPLORE_BONUS = .025f, QSCALE = 5.f, TAU = .01f;
// The world model predicts (next_obs - obs) / DSCALE so all output dimensions have a comparable scale.
inline float DSCALE[OBS] = {.27f, .27f, 1.f, 1.f, .14f, .14f, .14f, .14f, 1.f, 1.f};

struct Cfg {
  bool sparse = false;      // reward only at the goal
  bool heur = true;         // hand-made distance prior in the planner
  bool label = true;        // expose the regime id (T0/T1/T2) in the observation
  bool ctx = true;          // short-term memory (last velocity change + last action) for the world model
  bool oracle = false;      // diagnostic: planner uses the TRUE reward function on predicted states
  bool value = false;       // learned double-Q value as the planner's terminal value
  bool randomize = false;   // train on randomized dynamics instead of the 3 fixed regimes
  bool suite = false;       // final evaluation on the full transfer suite
  int trace = -1;            // eval-only: print per-episode behaviour for this scenario id
  bool clock_fix = false;    // planner advances the phase clock itself instead of trusting the model
  int lr_from = 0, lr_to = 0;  // --lr-decay A B: learning rate falls linearly from 1x at episode A to 0.1x at episode B (0 0 = off)
  std::vector<int> val_ids;  // --val-ids: structures used for best-snapshot validation (default: the final-evaluation ones)
  int best_from = 0;         // --best-from E: no validation before episode E (saves time when only the late snapshots matter)
  bool val_start = false;    // --val-start: validate the snapshot we resume from and keep it as a candidate
  float avg = 0.f;           // --avg tau: evaluate with an exponential moving average of the weights (time constant 1/tau gradient updates; 0 = off)
  bool rw_diag = false;      // --rw-diag (with --eval-only): does the reward head see failures (terminal -1 steps) and goals?
  bool tok_diag = false;     // --tok-diag (with --eval-only): next-token accuracy of the symbol model on fresh episodes
  float curious = 0.f;       // --curious BETA: intrinsic reward = predicted world-model error of the planned step (0 = off)
  bool blind = false;        // --blind: the agent never sees the game's reward or its end (self-exploration test); success is still counted
  int graph = 0;             // --graph NC: persistent transition graph over NC concepts; replay keeps one pinned exemplar per edge (0 = off)
  float graph_w = 0.f;       // --graph-w: weight of the graph value in the planner's terminal value
  int rcap = REPLAY_CAP;     // --rcap N: replay capacity (smaller values make forgetting visible)
  int hier = 0, hier_T = 20;  // --hier NC: manager over NC learned concepts, options last at most hier_T steps (0 = off)
  float hier_w = 4.f;        // --hier-w: weight of the low-level progress reward towards the chosen concept
  bool bottleneck = false;   // --bottleneck: only concepts that successful episodes pass through more often than failed ones may be sub-goals
  bool synth_lib = false;    // --synth-lib: also search formulas over a feature library (several dimensions, products, sin/cos)
  std::string synth_report;   // --synth-report FILE: write the unexplained dimensions with sample rows, for a person or a language model to propose formulas
  int synth_probe = 0;       // --synth-probe ID: collect random-policy transitions from scenario ID and print what the synthesis finds (no training)
  bool autocfg = false;      // --auto: no per-game flags; the agent probes the task with random moves and decides vq / rnd itself
  bool synth_diag = false;   // --synth-diag: print every rule / near miss found on the stored replay (with --eval-only --resume)
  bool synth = false;        // --synth: symbolic rule synthesis (v10)
  std::string rules;         // --rules FILE: externally supplied rule hypotheses, verified against experience
  int vq = 0;                // --vq K: discrete symbols, K bins per active dimension; the planner uses the next-token model (0 = off)
  int best_every = 0;        // keep the snapshot with the best validation score (every N episodes; 0 = off)
  int best_eps = 6;          // validation episodes per scenario
  int unroll = 0;            // multi-step world-model training over this many steps (0/1 = off)
  int unroll_batch = 8;
  float unroll_w = 1.f;
  int repeat = 1;            // macro-action length (planning over repeat*horizon primitive steps)
  std::vector<int> train_ids, final_ids;  // explicit scenario lists (--train-ids / --final-ids)
  bool show_phase = false;   // observation slot 9 = position in the wind cycle (sawtooth)
  bool show_wind = false;    // diagnostic: observation slot 9 carries the signed wind instead of its magnitude
  int wm_extra = 0;          // eval-only: extra world-model updates on the checkpoint's replay (diagnostic)
  bool ms_diag = false;      // print open-loop multi-step prediction error and exit (with --eval-only)
  bool oracle_value = false;  // diagnostic: planner terminal value from true geometry (key-door)
  bool rnet = false;        // reward predicted by a separate net on distance features
  float rw = REWARD_LOSS_W;  // loss weight of the reward output
  bool keydoor = false;     // train on the key-door task (collect the key, then reach the goal)
  bool final_keydoor = false;  // final evaluation only on the key-door scenarios
  bool final_hidden = false;  // final evaluation only on the hidden-goal scenarios (cheap, for many-seed experiments)
  bool hidden = false;      // train on the hidden-goal search task (goal visible only within the sense radius)
  float rnd_beta = 0.f;     // weight of the RND novelty bonus (0 = off)
  bool reservoir = false;   // learned-readout memory: fixed random leaky recurrent network over the history
  bool gmem = false;        // remember the goal position once seen and complete the hidden goal vector by dead reckoning
  float cover = 0.f;        // planner bonus per newly inspected map cell while the goal is hidden (0 = off)
  float pos_frac = 0.f;     // fraction of each training batch drawn from goal-reaching transitions
  float gamma = .94f;
  int horizon = 8, pop = 32, iters = 2, q_updates = 1;
};
inline Cfg cfg;

using Vec = std::array<float, OBS>;
using Out = std::array<float, OUT>;

inline float clampf(float x, float a, float b) { return std::max(a, std::min(b, x)); }
inline float sq(float x) { return x * x; }

// Rational tanh approximation (max abs error ~4e-7, verified by --selftest). ~4x faster than std::tanh.
inline float ftanh(float x) {
  x = std::max(-7.90531110763549805f, std::min(7.90531110763549805f, x));
  const float x2 = x * x;
  float p = x2 * -2.76076847742355e-16f + 2.00018790482477e-13f;
  p = x2 * p + -8.60467152213735e-11f;
  p = x2 * p + 5.12229709037114e-08f;
  p = x2 * p + 1.48572235717979e-05f;
  p = x2 * p + 6.37261928875436e-04f;
  p = x2 * p + 4.89352455891786e-03f;
  p = x * p;
  float q = x2 * 1.19825839466702e-06f + 1.18534705686654e-04f;
  q = x2 * q + 2.26843463243900e-03f;
  q = x2 * q + 4.89352518554385e-03f;
  return p / q;
}

// ---- short-term memory ------------------------------------------------------------------------
struct Ctx { float dv[CTX] = {0, 0}; int pa = -1; std::array<float, MEM> m{}; };
using In = std::array<float, CIN>;
inline In make_in(const Vec& s, const Ctx& c) {
  In x{};
  for (int i = 0; i < OBS; i++) x[i] = s[i];
  x[OBS] = c.dv[0]; x[OBS + 1] = c.dv[1];
  for (int i = 0; i < MEM; i++) x[OBS + CTX + i] = c.m[i];
  return x;
}

// Echo-state memory: a FIXED random leaky recurrent network driven by the observation/action history.
// Nothing here is task-specific; the networks that read it (world model, Q-function) learn what to extract.
// Units have log-uniform leak rates (0.01 .. 0.3) so some remember for ~100 steps and others for ~3.
struct Reservoir {
  std::array<float, MEM * MEM> wr{};
  std::array<float, MEM * (OBS + ACT)> wi{};
  std::array<float, MEM> alpha{};
  Reservoir() {
    std::mt19937 r(424242);  // fixed: the memory dynamics are part of the architecture, not of the run seed
    std::normal_distribution<float> n(0.f, 1.f);
    std::uniform_real_distribution<float> u(0.f, 1.f);
    const float rho = .9f, gain = .7f;  // spectral radius ~rho for a random matrix with variance rho^2/MEM
    for (auto& w : wr) w = n(r) * rho / std::sqrt((float)MEM);
    for (auto& w : wi) w = n(r) * gain;
    for (auto& a : alpha) a = std::exp(std::log(.01f) + (std::log(.3f) - std::log(.01f)) * u(r));
  }
  void step(const std::array<float, MEM>& m, const Vec& ns, int a, std::array<float, MEM>& out) const {
    for (int i = 0; i < MEM; i++) {
      float z = wi[i * (OBS + ACT) + OBS + a];
      for (int j = 0; j < MEM; j++) z += wr[i * MEM + j] * m[j];
      for (int k = 0; k < OBS; k++) z += wi[i * (OBS + ACT) + k] * ns[k];
      out[i] = (1.f - alpha[i]) * m[i] + alpha[i] * std::tanh(z);
    }
  }
};
inline const Reservoir& reservoir() { static const Reservoir R; return R; }

inline Ctx next_ctx(const Vec& s, const Vec& ns, int a, const Ctx& prev) {
  Ctx c;
  if (cfg.ctx) { c.dv[0] = ns[2] - s[2]; c.dv[1] = ns[3] - s[3]; c.pa = a; }
  if (cfg.reservoir) reservoir().step(prev.m, ns, a, c.m);
  return c;
}

// Episodic search memory: coarse grid of cells that were already inspected (inside the sense radius).
constexpr int GRID = 10;
struct Seen {
  std::array<uint8_t, GRID * GRID> c{};
  int mark(float x, float y, float r = .4f) {  // returns the number of newly inspected cells
    int n = 0;
    for (int gy = 0; gy < GRID; gy++)
      for (int gx = 0; gx < GRID; gx++) {
        uint8_t& cell = c[gy * GRID + gx];
        if (cell) continue;
        float cx = -1.f + 2.f * (gx + .5f) / GRID, cy = -1.f + 2.f * (gy + .5f) / GRID;
        if (std::hypot(cx - x, cy - y) <= r) { cell = 1; n++; }
      }
    return n;
  }
  int count() const { int n = 0; for (auto v : c) n += v; return n; }
};

// Goal memory: once the goal has been seen, remember its absolute position and keep the goal-relative
// observation filled in (the goal is static, the agent knows its own position).
struct GoalMem {
  bool have = false;
  float gx = 0, gy = 0;
  Vec fix(Vec o) {
    if (o[4] != 0.f || o[5] != 0.f) { have = true; gx = o[0] + 2.f * o[4]; gy = o[1] + 2.f * o[5]; }
    else if (have) { o[4] = clampf(gx - o[0], -2.f, 2.f) / 2.f; o[5] = clampf(gy - o[1], -2.f, 2.f) / 2.f; }
    return o;
  }
};

struct Transition {
  Vec s{}, ns{};
  Ctx c;
  int a = 0;
  float r = 0;
  bool done = false;  // goal reached (counts as success)
  bool term = false;  // episode ended in failure: no goal, but also no future value to bootstrap from (v19; before, the Q target looked past a failure)
  uint8_t task = 0;
};

template <class T> void wr(std::ostream& o, const T& v) { o.write(reinterpret_cast<const char*>(&v), sizeof(T)); }
template <class T> void rd(std::istream& i, T& v) { i.read(reinterpret_cast<char*>(&v), sizeof(T)); }

template <size_t N>
void adam_step(std::array<float, N>& p, std::array<float, N>& m, std::array<float, N>& v,
               const std::array<float, N>& g, float scale, float lr, float c1, float c2) {
  constexpr float B1 = .9f, B2 = .999f, EPS = 1e-8f;
  for (size_t i = 0; i < N; i++) {
    float gg = clampf(g[i] * scale, -5.f, 5.f);
    m[i] = B1 * m[i] + (1.f - B1) * gg;
    v[i] = B2 * v[i] + (1.f - B2) * gg * gg;
    p[i] -= lr * (m[i] / c1) / (std::sqrt(v[i] / c2) + EPS);
  }
}
template <size_t N> void put(std::ostream& o, const std::array<float, N>& a) { o.write(reinterpret_cast<const char*>(a.data()), sizeof(a)); }
template <size_t N> void get(std::istream& i, std::array<float, N>& a) { i.read(reinterpret_cast<char*>(a.data()), sizeof(a)); }

// ---------------------------------------------------------------------------------------------
// World model network: input = obs + memory (dense) + one-hot action + one-hot previous action.
// 2 tanh hidden layers, LINEAR output, MSE loss, mini-batch gradient accumulation + Adam.
// ---------------------------------------------------------------------------------------------
struct Grad {
  std::array<float, H1 * IN> w1{};
  std::array<float, H1> b1{};
  std::array<float, H2 * H1> w2{};
  std::array<float, H2> b2{};
  std::array<float, OUT * H2> w3{};
  std::array<float, OUT> b3{};
};

// Dot product with 8 independent partial sums: without -ffast-math the compiler may not reorder a float reduction, so the plain loop stays scalar;
// with eight lanes it vectorizes (about 4x faster on the forward pass, which is 97% of the run time). Summation order differs from v16, so results
// agree with older builds only up to rounding.
inline float dotf(const float* a, const float* b, int n) {
  float s[8] = {0, 0, 0, 0, 0, 0, 0, 0}; int i = 0;
  for (; i + 8 <= n; i += 8) for (int k = 0; k < 8; k++) s[k] += a[i + k] * b[i + k];
  float t = ((s[0] + s[4]) + (s[1] + s[5])) + ((s[2] + s[6]) + (s[3] + s[7]));
  for (; i < n; i++) t += a[i] * b[i];
  return t;
}

struct MLP {
  std::array<float, H1 * IN> w1{}, m1{}, v1{};
  std::array<float, H1> b1{}, mb1{}, vb1{};
  std::array<float, H2 * H1> w2{}, m2{}, v2{};
  std::array<float, H2> b2{}, mb2{}, vb2{};
  std::array<float, OUT * H2> w3{}, m3{}, v3{};
  std::array<float, OUT> b3{}, mb3{}, vb3{};
  uint64_t step = 0;

  // exponential moving average of the weights (not the optimizer state): this <- this + tau * (o - this)
  void blend(const MLP& o, float tau) {
    auto mix = [&](auto& a, const auto& b) { for (size_t i = 0; i < a.size(); i++) a[i] += tau * (b[i] - a[i]); };
    mix(w1, o.w1); mix(b1, o.b1); mix(w2, o.w2); mix(b2, o.b2); mix(w3, o.w3); mix(b3, o.b3);
  }

  MLP() = default;
  explicit MLP(uint32_t seed) {
    std::mt19937 rng(seed);
    auto init = [&](auto& w, int fan_in) {
      std::normal_distribution<float> d(0.f, std::sqrt(1.f / fan_in));
      for (auto& x : w) x = d(rng);
    };
    init(w1, IN); init(w2, H1); init(w3, H2);
    for (auto& x : w3) x *= .5f;
  }

  void layers(const In& x, int a, int pa, std::array<float, H1>& h1, std::array<float, H2>& h2, Out& y) const {
    for (int j = 0; j < H1; j++) {
      const float* w = &w1[j * IN];
      float z = b1[j] + w[CIN + a] + (pa >= 0 ? w[CIN + ACT + pa] : 0.f);
      z += dotf(w, x.data(), CIN);
      h1[j] = ftanh(z);
    }
    for (int j = 0; j < H2; j++) {
      const float* w = &w2[j * H1];
      float z = b2[j];
      z += dotf(w, h1.data(), H1);
      h2[j] = ftanh(z);
    }
    for (int k = 0; k < OUT; k++) {
      const float* w = &w3[k * H2];
      float z = b3[k];
      z += dotf(w, h2.data(), H2);
      y[k] = z;
    }
  }
  Out forward(const In& x, int a, int pa) const {
    std::array<float, H1> h1{};
    std::array<float, H2> h2{};
    Out y{};
    layers(x, a, pa, h1, h2, y);
    return y;
  }

  // Adds the gradient of one sample's loss to g. Returns that loss (weighted MSE / OUT).
  float accumulate(const In& x, int a, int pa, const Out& t, Grad& g) const {
    std::array<float, H1> h1{}, d1{};
    std::array<float, H2> h2{}, d2{};
    Out y{}, dy{};
    layers(x, a, pa, h1, h2, y);
    float loss = 0;
    for (int k = 0; k < OUT; k++) {
      float wk = (k == OBS) ? cfg.rw : 1.f, e = y[k] - t[k];
      loss += wk * e * e;
      dy[k] = 2.f * wk * e / OUT;
    }
    for (int k = 0; k < OUT; k++) {
      g.b3[k] += dy[k];
      float* gw = &g.w3[k * H2];
      for (int j = 0; j < H2; j++) gw[j] += dy[k] * h2[j];
    }
    for (int j = 0; j < H2; j++) {
      float z = 0;
      for (int k = 0; k < OUT; k++) z += w3[k * H2 + j] * dy[k];
      d2[j] = z * (1.f - h2[j] * h2[j]);
    }
    for (int j = 0; j < H2; j++) {
      g.b2[j] += d2[j];
      float* gw = &g.w2[j * H1];
      for (int i = 0; i < H1; i++) gw[i] += d2[j] * h1[i];
    }
    for (int i = 0; i < H1; i++) {
      float z = 0;
      for (int j = 0; j < H2; j++) z += w2[j * H1 + i] * d2[j];
      d1[i] = z * (1.f - h1[i] * h1[i]);
    }
    for (int j = 0; j < H1; j++) {
      g.b1[j] += d1[j];
      float* gw = &g.w1[j * IN];
      for (int i = 0; i < CIN; i++) gw[i] += d1[j] * x[i];
      gw[CIN + a] += d1[j];
      if (pa >= 0) gw[CIN + ACT + pa] += d1[j];
    }
    return loss / OUT;
  }

  // Backpropagates a given output gradient dy (activations from layers()); also returns d(loss)/d(input x).
  void backprop(const In& x, int a, int pa, const std::array<float, H1>& h1, const std::array<float, H2>& h2, const Out& dy, Grad& g, In& dx) const {
    std::array<float, H1> d1{};
    std::array<float, H2> d2{};
    for (int k = 0; k < OUT; k++) {
      g.b3[k] += dy[k];
      float* gw = &g.w3[k * H2];
      for (int j = 0; j < H2; j++) gw[j] += dy[k] * h2[j];
    }
    for (int j = 0; j < H2; j++) {
      float z = 0;
      for (int k = 0; k < OUT; k++) z += w3[k * H2 + j] * dy[k];
      d2[j] = z * (1.f - h2[j] * h2[j]);
    }
    for (int j = 0; j < H2; j++) {
      g.b2[j] += d2[j];
      float* gw = &g.w2[j * H1];
      for (int i = 0; i < H1; i++) gw[i] += d2[j] * h1[i];
    }
    for (int i = 0; i < H1; i++) {
      float z = 0;
      for (int j = 0; j < H2; j++) z += w2[j * H1 + i] * d2[j];
      d1[i] = z * (1.f - h1[i] * h1[i]);
    }
    dx.fill(0.f);
    for (int j = 0; j < H1; j++) {
      g.b1[j] += d1[j];
      float* gw = &g.w1[j * IN];
      const float* w = &w1[j * IN];
      for (int i = 0; i < CIN; i++) { gw[i] += d1[j] * x[i]; dx[i] += d1[j] * w[i]; }
      gw[CIN + a] += d1[j];
      if (pa >= 0) gw[CIN + ACT + pa] += d1[j];
    }
  }

  void apply(const Grad& g, float scale, float lr) {
    step++;
    float c1 = 1.f - (float)std::pow(.9, (double)step), c2 = 1.f - (float)std::pow(.999, (double)step);
    adam_step(w1, m1, v1, g.w1, scale, lr, c1, c2);
    adam_step(b1, mb1, vb1, g.b1, scale, lr, c1, c2);
    adam_step(w2, m2, v2, g.w2, scale, lr, c1, c2);
    adam_step(b2, mb2, vb2, g.b2, scale, lr, c1, c2);
    adam_step(w3, m3, v3, g.w3, scale, lr, c1, c2);
    adam_step(b3, mb3, vb3, g.b3, scale, lr, c1, c2);
  }
  void save(std::ostream& o) const {
    put(o, w1); put(o, m1); put(o, v1); put(o, b1); put(o, mb1); put(o, vb1);
    put(o, w2); put(o, m2); put(o, v2); put(o, b2); put(o, mb2); put(o, vb2);
    put(o, w3); put(o, m3); put(o, v3); put(o, b3); put(o, mb3); put(o, vb3);
    wr(o, step);
  }
  bool load(std::istream& i) {
    get(i, w1); get(i, m1); get(i, v1); get(i, b1); get(i, mb1); get(i, vb1);
    get(i, w2); get(i, m2); get(i, v2); get(i, b2); get(i, mb2); get(i, vb2);
    get(i, w3); get(i, m3); get(i, v3); get(i, b3); get(i, mb3); get(i, vb3);
    rd(i, step);
    return bool(i);
  }
};

// ---------------------------------------------------------------------------------------------
// Generic dense network (2 tanh hidden layers, linear output). Used for RND and the Q-function.
// ---------------------------------------------------------------------------------------------
template <int NI, int N1, int N2, int NO>
struct DenseNet {
  using InT = std::array<float, NI>;
  using OutT = std::array<float, NO>;
  struct Act { std::array<float, N1> h1{}; std::array<float, N2> h2{}; };
  struct Grad {
    std::array<float, N1 * NI> w1{};
    std::array<float, N1> b1{};
    std::array<float, N2 * N1> w2{};
    std::array<float, N2> b2{};
    std::array<float, NO * N2> w3{};
    std::array<float, NO> b3{};
  };
  std::array<float, N1 * NI> w1{}, m1{}, v1{};
  std::array<float, N1> b1{}, mb1{}, vb1{};
  std::array<float, N2 * N1> w2{}, m2{}, v2{};
  std::array<float, N2> b2{}, mb2{}, vb2{};
  std::array<float, NO * N2> w3{}, m3{}, v3{};
  std::array<float, NO> b3{}, mb3{}, vb3{};
  uint64_t step = 0;

  void init(uint32_t seed, float out_scale) {
    std::mt19937 rng(seed);
    auto fill = [&](auto& w, int fan_in, float sc) {
      std::normal_distribution<float> d(0.f, sc * std::sqrt(1.f / fan_in));
      for (auto& x : w) x = d(rng);
    };
    fill(w1, NI, 1.f); fill(w2, N1, 1.f); fill(w3, N2, out_scale);
  }
  void forward(const InT& x, Act& a, OutT& y) const {
    for (int j = 0; j < N1; j++) {
      const float* w = &w1[j * NI];
      float z = b1[j];
      for (int i = 0; i < NI; i++) z += w[i] * x[i];
      a.h1[j] = ftanh(z);
    }
    for (int j = 0; j < N2; j++) {
      const float* w = &w2[j * N1];
      float z = b2[j];
      for (int i = 0; i < N1; i++) z += w[i] * a.h1[i];
      a.h2[j] = ftanh(z);
    }
    for (int k = 0; k < NO; k++) {
      const float* w = &w3[k * N2];
      float z = b3[k];
      for (int j = 0; j < N2; j++) z += w[j] * a.h2[j];
      y[k] = z;
    }
  }
  OutT forward(const InT& x) const {
    Act a;
    OutT y{};
    forward(x, a, y);
    return y;
  }
  // Accumulates d(loss)/d(params) given dy = d(loss)/d(output).
  void backward(const InT& x, const Act& a, const OutT& dy, Grad& g) const {
    std::array<float, N2> d2{};
    std::array<float, N1> d1{};
    for (int k = 0; k < NO; k++) {
      g.b3[k] += dy[k];
      float* gw = &g.w3[k * N2];
      for (int j = 0; j < N2; j++) gw[j] += dy[k] * a.h2[j];
    }
    for (int j = 0; j < N2; j++) {
      float z = 0;
      for (int k = 0; k < NO; k++) z += w3[k * N2 + j] * dy[k];
      d2[j] = z * (1.f - a.h2[j] * a.h2[j]);
    }
    for (int j = 0; j < N2; j++) {
      g.b2[j] += d2[j];
      float* gw = &g.w2[j * N1];
      for (int i = 0; i < N1; i++) gw[i] += d2[j] * a.h1[i];
    }
    for (int i = 0; i < N1; i++) {
      float z = 0;
      for (int j = 0; j < N2; j++) z += w2[j * N1 + i] * d2[j];
      d1[i] = z * (1.f - a.h1[i] * a.h1[i]);
    }
    for (int j = 0; j < N1; j++) {
      g.b1[j] += d1[j];
      float* gw = &g.w1[j * NI];
      for (int i = 0; i < NI; i++) gw[i] += d1[j] * x[i];
    }
  }
  void apply(const Grad& g, float scale, float lr) {
    step++;
    float c1 = 1.f - (float)std::pow(.9, (double)step), c2 = 1.f - (float)std::pow(.999, (double)step);
    adam_step(w1, m1, v1, g.w1, scale, lr, c1, c2);
    adam_step(b1, mb1, vb1, g.b1, scale, lr, c1, c2);
    adam_step(w2, m2, v2, g.w2, scale, lr, c1, c2);
    adam_step(b2, mb2, vb2, g.b2, scale, lr, c1, c2);
    adam_step(w3, m3, v3, g.w3, scale, lr, c1, c2);
    adam_step(b3, mb3, vb3, g.b3, scale, lr, c1, c2);
  }
  // Polyak averaging of the weights (not the optimizer state): this <- tau * o + (1 - tau) * this
  void blend(const DenseNet& o, float tau) {
    auto mix = [&](auto& a, const auto& b) { for (size_t i = 0; i < a.size(); i++) a[i] += tau * (b[i] - a[i]); };
    mix(w1, o.w1); mix(b1, o.b1); mix(w2, o.w2); mix(b2, o.b2); mix(w3, o.w3); mix(b3, o.b3);
  }
  void save(std::ostream& o) const {
    put(o, w1); put(o, m1); put(o, v1); put(o, b1); put(o, mb1); put(o, vb1);
    put(o, w2); put(o, m2); put(o, v2); put(o, b2); put(o, mb2); put(o, vb2);
    put(o, w3); put(o, m3); put(o, v3); put(o, b3); put(o, mb3); put(o, vb3);
    wr(o, step);
  }
  bool load(std::istream& i) {
    get(i, w1); get(i, m1); get(i, v1); get(i, b1); get(i, mb1); get(i, vb1);
    get(i, w2); get(i, m2); get(i, v2); get(i, b2); get(i, mb2); get(i, vb2);
    get(i, w3); get(i, m3); get(i, v3); get(i, b3); get(i, mb3); get(i, vb3);
    rd(i, step);
    return bool(i);
  }
};
using RndNet = DenseNet<OBS, RH, RH, RO>;
using QNet = DenseNet<CIN, QH, QH, ACT>;
constexpr int RF = 6;  // reward-net features
using RwNet = DenseNet<RF, 24, 24, 1>;

// ---------------------------------------------------------------------------------------------
// Environment: partially observed 2D navigation with hidden wind that flips sign periodically.
// A scenario = dynamics (friction, thrust, wind, flip period) + map type.
// ---------------------------------------------------------------------------------------------
enum class MapMode { Random, Blocker, NearGoal };
struct Dyn { float f = .76f, sp = .105f, wx = .075f, wy = -.045f; int flip = 100; };
struct Scn { const char* name; int lab; Dyn d; MapMode m; float sense = 0.f; bool keydoor = false, invert = false, gmove = false, chain = false; int game = 0; };  // sense > 0: goal hidden beyond this distance, sparse reward

constexpr int N_SEEN = 3, N_SCN = 21;
// ---- procedural task family ------------------------------------------------------------------------------------
// Index i deterministically defines one task STRUCTURE: how many keys must be collected before the goal (0, 1 or 2), whether the controls
// mirror after the key, whether the goal drifts, whether the goal is hidden (search), and its own physics (friction, thrust, wind, flip
// period). Wind is always weaker than thrust/1.3, so every generated task is physically solvable (T2 is not).
// Scenario ids >= GEN_BASE are generated tasks: id = GEN_BASE + index.
constexpr int GEN_BASE = 100, GEN_MAX = 2000;
inline Scn gen_scenario(int index) {
  static std::map<int, std::string> names;  // stable storage for the c_str() in Scn::name (one entry per index)
  std::mt19937 r(0x9E3779B1u ^ (uint32_t)(index * 2654435761u));
  std::uniform_real_distribution<float> u(0.f, 1.f);
  auto U = [&](float a, float b) { return a + (b - a) * u(r); };
  Dyn d;
  d.f = U(.68f, .84f);
  d.sp = U(.085f, .125f);
  const float wmax = d.sp / 1.3f;
  d.wx = (u(r) < .5f ? -1.f : 1.f) * U(.02f, wmax);
  d.wy = (u(r) < .5f ? -1.f : 1.f) * U(.01f, std::min(.04f, wmax));
  d.flip = 80 + (int)(u(r) * 50.f);
  const float pk = u(r);
  const int nkeys = pk < .30f ? 0 : pk < .65f ? 1 : 2;
  Scn s{"", -1, d, MapMode::Random};
  if (nkeys == 1) { s.keydoor = true; s.invert = u(r) < .35f; }
  else if (nkeys == 2) s.chain = true;
  s.gmove = u(r) < .25f;
  if (nkeys == 0 && !s.gmove && u(r) < .25f) s.sense = .4f;
  std::string nm = "G" + std::to_string(index) + (nkeys == 2 ? "c" : nkeys == 1 ? (s.invert ? "v" : "k") : (s.sense > 0.f ? "h" : "p")) + (s.gmove ? "m" : "");
  s.name = names.emplace(index, nm).first->second.c_str();
  return s;
}

inline Scn scenario(int id) {
  if (id == GAME_BASE + 200) return Scn{"External", -1, Dyn{}, MapMode::Random, 0.f, false, false, false, false, 100};
  if (id >= HELD_BASE && id < HELD_BASE + HELD_N) {
    static const char* const hn[HELD_N] = {"Lights-cross", "Lights-diagonal", "Lights-cyclic", "Lights-rowcol", "Mod61", "Mod89", "Mod113", "Mod127", "Grid-walls", "Grid-teleports",
                                           "Sort4", "Sort5", "Hanoi3", "Hanoi4", "Registers16", "Registers12", "Lock", "Lock-coupled", "Ring64", "Perm64"};
    return Scn{hn[id - HELD_BASE], -1, Dyn{}, MapMode::Random, 0.f, false, false, false, false, 10 + id - HELD_BASE};
  }
  if (id >= GAME_BASE) {
    static const char* const gn[] = {"MountainCar", "NumberChain", "Pendulum", "Switches", "ModChain"};
    return Scn{gn[std::min(id - GAME_BASE, 4)], -1, Dyn{}, MapMode::Random, 0.f, false, false, false, false, id - GAME_BASE + 1};
  }
  if (id >= GEN_BASE) return gen_scenario(id - GEN_BASE);
  const Dyn t0{.76f, .105f, .075f, -.045f, 100}, t1{.76f, .085f, -.075f, .045f, 100}, t2{.76f, .12f, .13f, -.025f, 100};
  switch (id) {
    case 0: return {"T0", 0, t0, MapMode::Random};
    case 1: return {"T1", 1, t1, MapMode::Random};
    case 2: return {"T2", 2, t2, MapMode::Random};
    case 3: return {"N0-heavy", -1, Dyn{.84f, .095f, .09f, -.03f, 100}, MapMode::Random};
    case 4: return {"N1-light", -1, Dyn{.68f, .125f, -.06f, .05f, 100}, MapMode::Random};
    case 5: return {"N2-gusty", -1, Dyn{.76f, .105f, .15f, -.05f, 50}, MapMode::Random};
    case 6: return {"M0-blocker", 0, t0, MapMode::Blocker};
    case 7: return {"M1-guarded", 2, t2, MapMode::NearGoal};
    case 9: return {"H0-search", -1, t0, MapMode::Random, .4f};
    case 10: return {"H1-search", -1, t1, MapMode::Random, .4f};
    case 11: return {"H2-search", -1, t2, MapMode::Random, .4f};
    case 12: return {"K0-keydoor", -1, t0, MapMode::Random, 0.f, true};
    case 13: return {"K1-keydoor", -1, t1, MapMode::Random, 0.f, true};
    case 14: return {"K2-keydoor", -1, t2, MapMode::Random, 0.f, true};
    case 15: return {"C0-chain", -1, t0, MapMode::Random, 0.f, false, false, false, true};     // two keys in order, then the goal
    case 16: return {"C1-chain", -1, t1, MapMode::Random, 0.f, false, false, false, true};
    case 17: return {"V0-invert", -1, t0, MapMode::Random, 0.f, true, true};                  // key-door, but the controls mirror after the key
    case 18: return {"V1-invert", -1, t1, MapMode::Random, 0.f, true, true};
    case 19: return {"G0-moving", -1, t0, MapMode::Random, 0.f, false, false, true};         // the goal drifts
    case 20: return {"G1-moving", -1, t1, MapMode::Random, 0.f, false, false, true};
    default: return {"E0-extrap", -1, Dyn{.90f, .07f, .12f, .04f, 100}, MapMode::Random};
  }
}
inline Dyn random_dyn(std::mt19937& r) {
  std::uniform_real_distribution<float> u01(0.f, 1.f);
  auto U = [&](float a, float b) { return a + (b - a) * u01(r); };
  Dyn d;
  d.f = U(.66f, .86f);
  d.sp = U(.08f, .13f);
  d.wx = (u01(r) < .5f ? -1.f : 1.f) * U(.04f, .16f);
  d.wy = (u01(r) < .5f ? -1.f : 1.f) * U(.01f, .06f);
  d.flip = 60 + (int)(u01(r) * 80.f);
  return d;
}

struct Reward { float r; bool done, hit; };
inline Reward reward_fn(float d, float od, bool sparse) {
  bool hit = od < .14f, done = d < .085f && !hit;
  float r = sparse ? -.01f + (done ? 4.f : 0.f)
                   : -.035f - .55f * d - .20f * std::max(0.f, .23f - od) + (done ? 4.f : 0.f) - (hit ? 1.2f : 0.f);
  return {r, done, hit};
}

// ---- Held-out family (scenario ids 3100..3119 = game 10..29): 20 fixed puzzles of kinds the agent was never tuned on.
// Each puzzle is a deterministic graph over n states with 9 actions (structure seeded by the kind, identical in every episode); an episode draws a
// start (and, for some kinds, a goal) at random. Reward as in the other games: -0.01 per step, +4 and done at the goal. Oracle: BFS on the table.
// ---- Embedded helper scripts, so this one file is everything: `--write-bridge FILE` / `--write-llm-hints FILE` write them out and exit. ----
static const char* const BRIDGE_PY = R"NEXUSPY(#!/usr/bin/env python3
"""Bridge between nexus_cognitive_engine and a Gymnasium environment (or a built-in test CartPole).

Usage (run by the engine itself):  ./nexus --ext "python3 nexus_gym_bridge.py CartPole-v1" --train-ids 3200 --final-ids 3200 --auto
Environments: any Gymnasium id with a Box observation of <= 10 numbers and a Discrete or 1-D Box action; `fake:CartPole` needs no
gymnasium (pure-Python CartPole with the Gym equations; used to test the protocol).

Protocol (one text line per request / reply on stdin / stdout):
  start      -> "ready <obs_dim> <max_steps>"
  reset      -> "S <obs...>"
  step <a>   -> "S <obs...> <fail> <goal> <truncated>"       (a = 0..8; mapped to the env's actions, see action_map)
  quit
Observations are scaled to [-1, 1] (finite bounds linearly, unbounded dimensions with tanh(x/2)).
The env's own reward is NOT used. The engine's reward is -0.01 per step, +4 at a goal, -1 on failure; what counts as goal / failure is
defined per environment in RULES below (CartPole: surviving the step limit is the goal, a fall is a failure)."""
import math, random, sys


class FakeCartPole:  # Gym CartPole-v1 equations (Euler integration, tau .02), 4 observations, 2 actions
    max_steps = 200

    def __init__(self, seed=0):
        self.rng = random.Random(seed)
        self.n_actions, self.dim = 2, 4

    def reset(self):
        self.s = [self.rng.uniform(-.05, .05) for _ in range(4)]
        self.t = 0
        return list(self.s)

    def step(self, a):
        x, xd, th, thd = self.s
        force = 10.0 if a == 1 else -10.0
        ct, st = math.cos(th), math.sin(th)
        temp = (force + .05 * thd * thd * st) / 1.1
        thacc = (9.8 * st - ct * temp) / (.5 * (4.0 / 3.0 - .1 * ct * ct / 1.1))
        xacc = temp - .05 * thacc * ct / 1.1
        x, xd, th, thd = x + .02 * xd, xd + .02 * xacc, th + .02 * thd, thd + .02 * thacc
        self.s = [x, xd, th, thd]
        self.t += 1
        terminated = abs(x) > 2.4 or abs(th) > 12 * 2 * math.pi / 360
        return list(self.s), terminated, self.t >= self.max_steps

    def scale(self, o):
        return [max(-1, min(1, o[0] / 2.4)), math.tanh(o[1] / 2), max(-1, min(1, o[2] / .2095)), math.tanh(o[3] / 2)]


# environment id -> (what a terminated episode means, extra goal test on the raw observation)
RULES = {
    "CartPole-v1": "survive",       # terminated = fell (failure); truncated at the step limit = success
    "MountainCar-v0": "terminal",   # terminated = reached the flag (goal)
    "Acrobot-v1": "terminal",       # terminated = tip above the line (goal)
    "LunarLander-v3": "terminal",   # terminated = landed or crashed: treated as a goal only if the env's reward at that step is > 0
    "Pendulum-v1": "upright",       # goal when cos(theta) > .98 and |omega| < 1
}


def make(name, seed):
    if name.startswith("fake:"):
        env = FakeCartPole(seed)
        return env, "survive", env.dim, env.max_steps, lambda a: a % 2, env.scale
    import gymnasium as gym
    import numpy as np
    env = gym.make(name)
    env.reset(seed=seed)
    sp, asp = env.observation_space, env.action_space
    dim = int(np.prod(sp.shape))
    if dim > 10:
        raise SystemExit("observation has %d numbers, the engine takes at most 10" % dim)
    lo, hi = np.array(sp.low, dtype=float).ravel(), np.array(sp.high, dtype=float).ravel()

    def scale(o):
        o = np.array(o, dtype=float).ravel()
        out = []
        for i in range(dim):
            if np.isfinite(lo[i]) and np.isfinite(hi[i]) and hi[i] > lo[i] and hi[i] - lo[i] < 1e3:
                out.append(float(2 * (o[i] - lo[i]) / (hi[i] - lo[i]) - 1))
            else:
                out.append(float(math.tanh(o[i] / 2)))
        return out

    if hasattr(asp, "n"):
        amap = lambda a: a % int(asp.n)
    else:  # 1-D Box: nine evenly spaced torques
        amap = lambda a: np.array([(a - 4) / 4.0 * float(asp.high[0])], dtype=np.float32)
    steps = env.spec.max_episode_steps or 200
    return env, RULES.get(name, "terminal"), dim, int(steps), amap, scale


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "fake:CartPole"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    env, rule, dim, steps, amap, scale = make(name, seed)
    fake = name.startswith("fake:")
    last = None
    print("ready %d %d" % (dim, steps), flush=True)
    for line in sys.stdin:
        w = line.split()
        if not w:
            continue
        if w[0] == "quit":
            break
        if w[0] == "reset":
            o = env.reset() if fake else env.reset()[0]
            last = o
            print("S " + " ".join("%.6f" % v for v in scale(o)), flush=True)
        elif w[0] == "step":
            a = amap(int(w[1]))
            if fake:
                o, term, trunc = env.step(a)
                rew = 0.0
            else:
                o, rew, term, trunc, _ = env.step(a)
            last = o
            fail = goal = 0
            if rule == "survive":
                fail, goal = int(term), int(trunc and not term)
            elif rule == "upright":
                goal = int(float(o[0]) > .98 and abs(float(o[2])) < 1.0)
            else:
                goal = int(term and (name != "LunarLander-v3" or rew > 0)); fail = int(term and not goal)
            print("S " + " ".join("%.6f" % v for v in scale(o)) + " %d %d %d" % (fail, goal, int(trunc)), flush=True)


if __name__ == "__main__":
    main()
)NEXUSPY";
static const char* const LLM_HINTS_PY = R"NEXUSPY(#!/usr/bin/env python3
"""Language-model hints for the agent (the engine itself contains no language model).

  prompt REPORT [DESCRIPTION]   -> prints a prompt for any LLM: what the engine could not explain (from --synth-report) plus the rule syntax
  check  REPLY RULES_OUT        -> keeps only well-formed hypothesis lines from the LLM's reply and writes the file for  --rules RULES_OUT

The engine then CONFIRMS or REFUTES every hypothesis on its own experience (>= 99% of all samples of that action), so a wrong guess costs
nothing; only confirmed rules are used. Nothing here calls a network service: paste the prompt into a model of your choice."""
import re, sys

SYNTAX = """Reply with hypotheses, one per line, nothing else.
 affine:   <action> <dim> <alpha> <beta>            meaning  s'[dim] = alpha*s[dim] + beta  when the action is taken (all numbers in the normalised units)
 formula:  f <action> <dim> = c + c1*s2 + c2*sin(w*s0) + c3*cos(w*s0) ...  [| when <expr> < c -> v]
           terms may use s0..s9, sin(), cos(), products and the constant c; the optional clause replaces the value by v when <expr> < c.
Lines starting with # are comments."""

NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
AFFINE = re.compile(r"^\s*(\d+)\s+(\d+)\s+(%s)\s+(%s)\s*(#.*)?$" % (NUM, NUM))
FORMULA = re.compile(r"^\s*f\s+(\d+)\s+(\d+)\s*=\s*([-+*/().,\ \dsincoe^|<>\w]+)$")
ALLOWED = re.compile(r"^[-+*/().,\s\d]*(?:(?:s\d|sin|cos|when|e|E)[-+*/().,\s\d<>|]*)*$")


def prompt(report, description=""):
    body = open(report).read()
    return ("I have a learning agent in a simulated world. It could not explain the following state dimensions with simple rules.\n"
            "%s\n\n--- what the agent observed ---\n%s\n\n%s\n" % (("World description: " + description) if description else "", body, SYNTAX))


def check(reply, out):
    keep, dropped = [], 0
    for line in open(reply):
        line = line.rstrip("\n")
        if not line.strip() or line.strip().startswith("#"):
            continue
        ok = False
        if AFFINE.match(line):
            ok = True
        else:
            m = FORMULA.match(line)
            if m and line.count("(") == line.count(")") and "__" not in line and ";" not in line:
                ok = True
        if ok:
            keep.append(line)
        else:
            dropped += 1
    with open(out, "w") as f:
        f.write("# hypotheses from a language model, syntax-checked by nexus_llm_hints.py; the engine confirms or refutes each on its own data\n")
        f.write("\n".join(keep) + "\n")
    print("kept %d hypotheses, dropped %d malformed lines" % (len(keep), dropped))


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "prompt":
        print(prompt(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else ""))
    elif len(sys.argv) == 4 and sys.argv[1] == "check":
        check(sys.argv[2], sys.argv[3])
    else:
        print(__doc__)
)NEXUSPY";
inline void write_text_file(const std::string& path, const char* text) {
  std::ofstream f(path);
  if (!f) throw std::runtime_error("cannot write " + path);
  f << text;
}

struct Puzzle {
  int kind = -1, n = 0, nd = 0, ng = 0, B = 2, GB = 2;  // states, state digits, goal digits (0 = fixed goal), digit bases
  bool gsingle = false;                                 // goal shown as one scaled number instead of digits
  std::vector<std::array<int, 9>> nx;
  std::vector<std::array<uint8_t, 9>> dg;
  int goal = 0;
  std::vector<int> goals;  // candidate goals (random-goal kinds)
  const char* name = "";
  static int dist_first(const std::vector<std::array<int, 9>>& nx, int s, int g, int* dist = nullptr) {  // BFS: first action on a shortest path
    const int n = (int)nx.size(); std::vector<int> first(n, -2), d(n, 0), q; q.push_back(s); first[s] = -1;
    for (size_t h = 0; h < q.size(); h++) {
      const int v = q[h];
      if (v == g) break;
      for (int a = 0; a < 9; a++) { const int u = nx[v][a]; if (first[u] == -2) { first[u] = v == s ? a : first[v]; d[u] = d[v] + 1; q.push_back(u); } }
    }
    if (first[g] == -2) { if (dist) *dist = -1; return 0; }
    if (dist) *dist = d[g];
    return std::max(0, first[g]);
  }
  void digits_base(int base, int count) {  // dg[s] = base-`base` digits of s
    nd = count; B = base; dg.assign(n, {});
    for (int s = 0; s < n; s++) { int v = s; for (int i = 0; i < count; i++) { dg[s][i] = (uint8_t)(v % base); v /= base; } }
  }
  void build(int k) {
    kind = k; std::mt19937 r(7919u * (uint32_t)k + 13u); ng = 0; gsingle = false; goals.clear();
    auto rnd = [&](int m) { return (int)(r() % (uint32_t)m); };
    if (k < 4) {  // lights on 9 cells, goal all on; 4 variants of which cells a press toggles
      static const char* const nm[] = {"Lights-cross", "Lights-diagonal", "Lights-cyclic", "Lights-rowcol"};
      name = nm[k]; n = 512; digits_base(2, 9); goal = 511; nx.assign(n, {});
      for (int p = 0; p < 9; p++) {
        int m = 0; const int px = p % 3, py = p / 3;
        for (int c = 0; c < 9; c++) {
          const int cx = c % 3, cy = c / 3; bool t = false;
          if (k == 0) t = c == p || (std::abs(cx - px) + std::abs(cy - py) == 1);
          if (k == 1) t = c == p || (std::abs(cx - px) == 1 && std::abs(cy - py) == 1);
          if (k == 2) t = c == p || c == (p + 1) % 9;
          if (k == 3) t = cx == px || cy == py;
          if (t) m |= 1 << c;
        }
        for (int s = 0; s < n; s++) nx[s][p] = s ^ m;
      }
      for (int s = 0; s < n; s++) { int d; dist_first(nx, s, goal, &d); if (d >= 3 && d <= 12) goals.push_back(s); }  // goals[] is the pool of START states here
    } else if (k < 8) {  // modular chains with random operations; goal drawn per episode
      static const int Ms[] = {61, 89, 113, 127}; static const char* const nm[] = {"Mod61", "Mod89", "Mod113", "Mod127"};
      name = nm[k - 4]; n = Ms[k - 4]; nx.assign(n, {}); ng = 1; gsingle = true;
      dg.assign(n, {}); nd = 1; B = n; for (int s = 0; s < n; s++) dg[s][0] = (uint8_t)s;
      for (int a = 0; a < 9; a++) {
        const int t = rnd(4), c = 1 + rnd(n - 1), e = rnd(n);
        for (int s = 0; s < n; s++) nx[s][a] = t == 0 ? (s + c) % n : t == 1 ? (s * (c % 5 + 2) + e) % n : t == 2 ? ((s - c) % n + n) % n : (s * (c % 4 + 2)) % n;
      }
      GB = n; for (int s = 0; s < n; s++) goals.push_back(s);
    } else if (k < 10) {  // 7x7 grid: k=8 walls and diagonal moves, k=9 teleport pads
      name = k == 8 ? "Grid-walls" : "Grid-teleports"; const int W = 7; n = W * W; nx.assign(n, {}); dg.assign(n, {}); nd = 2; B = W; ng = 2; GB = W;
      std::vector<int> wall(n, 0), pad(n, -1);
      if (k == 8) for (int i = 0; i < 10; i++) wall[rnd(n)] = 1;
      else for (int i = 0; i < 3; i++) { int a = rnd(n), b = rnd(n); if (a != b && pad[a] < 0 && pad[b] < 0) { pad[a] = b; pad[b] = a; } }
      static const int DX[9] = {1, -1, 0, 0, 1, -1, 1, -1, 0}, DY[9] = {0, 0, 1, -1, 1, 1, -1, -1, 0};
      for (int s = 0; s < n; s++) {
        dg[s][0] = (uint8_t)(s % W); dg[s][1] = (uint8_t)(s / W);
        for (int a = 0; a < 9; a++) {
          int x = s % W + DX[a], y = s / W + DY[a]; int t = s;
          const bool diag_ok = k == 8 || a < 4 || a == 8;
          if (diag_ok && x >= 0 && x < W && y >= 0 && y < W && !wall[y * W + x] && !(k == 9 && a >= 4 && a < 8)) { t = y * W + x; if (k == 9 && pad[t] >= 0) t = pad[t]; }
          nx[s][a] = t;
        }
      }
      for (int s = 0; s < n; s++) if (!wall[s]) goals.push_back(s);
    } else if (k < 12) {  // permutation puzzles: adjacent swaps, rotations, reversal of the first three; goal = sorted
      const int m = k == 10 ? 4 : 5; name = k == 10 ? "Sort4" : "Sort5";
      std::vector<std::vector<int>> perms; std::vector<int> p(m); for (int i = 0; i < m; i++) p[i] = i;
      do perms.push_back(p); while (std::next_permutation(p.begin(), p.end()));
      n = (int)perms.size(); nx.assign(n, {}); dg.assign(n, {}); nd = m; B = m; goal = 0;
      std::map<std::vector<int>, int> id; for (int s = 0; s < n; s++) id[perms[s]] = s;
      for (int s = 0; s < n; s++) {
        for (int i = 0; i < m; i++) dg[s][i] = (uint8_t)perms[s][i];
        for (int a = 0; a < 9; a++) {
          std::vector<int> q = perms[s];
          if (a < m - 1) std::swap(q[a], q[a + 1]);
          else if (a == m - 1) std::rotate(q.begin(), q.begin() + 1, q.end());
          else if (a == m) std::rotate(q.begin(), q.end() - 1, q.end());
          else if (a == m + 1) std::reverse(q.begin(), q.begin() + 3);
          nx[s][a] = id[q];
        }
      }
      for (int s = 1; s < n; s++) goals.push_back(s);
    } else if (k < 14) {  // Tower of Hanoi with 3 / 4 discs: state = peg of each disc; action = move from peg i to peg j (6 moves) + 3 no-ops
      const int m = k == 12 ? 3 : 4; name = k == 12 ? "Hanoi3" : "Hanoi4"; n = 1; for (int i = 0; i < m; i++) n *= 3;
      digits_base(3, m); nx.assign(n, {}); goal = n - 1;  // all discs on peg 2
      static const int FR[6] = {0, 0, 1, 1, 2, 2}, TO[6] = {1, 2, 0, 2, 0, 1};
      for (int s = 0; s < n; s++) for (int a = 0; a < 9; a++) {
        int t = s;
        if (a < 6) {
          int top = -1;  // smallest disc (index 0) on the source peg; legal if no smaller disc on the target
          for (int i = 0; i < m; i++) if (dg[s][i] == FR[a]) { top = i; break; }
          bool ok = top >= 0; if (ok) for (int i = 0; i < top; i++) if (dg[s][i] == TO[a]) ok = false;
          if (ok) { int pw = 1; for (int i = 0; i < top; i++) pw *= 3; t = s + (TO[a] - FR[a]) * pw; }
        }
        nx[s][a] = t;
      }
      for (int s = 0; s < n - 1; s++) goals.push_back(s);
    } else if (k < 16) {  // two registers mod M with arithmetic ops; goal pair drawn per episode
      const int M = k == 14 ? 16 : 12; name = k == 14 ? "Registers16" : "Registers12"; n = M * M; digits_base(M, 2); B = M; ng = 2; GB = M; nx.assign(n, {});
      for (int s = 0; s < n; s++) {
        const int a0 = s % M, b0 = s / M;
        for (int a = 0; a < 9; a++) {
          int x = a0, y = b0;
          if (k == 14) switch (a) { case 0: x = (x + 1) % M; break; case 1: y = (y + 1) % M; break; case 2: x = (x + y) % M; break; case 3: y = (y + x) % M; break; case 4: std::swap(x, y); break; case 5: x ^= y; break; case 6: x = (x * 2) % M; break; case 7: y = (y * 2) % M; break; default: x = ((x - y) % M + M) % M; }
          else switch (a) { case 0: x = (x + 3) % M; break; case 1: y = (y + 5) % M; break; case 2: x = ((x - y) % M + M) % M; break; case 3: y = ((y - x) % M + M) % M; break; case 4: x = (x * 3) % M; break; case 5: y = (y * 3) % M; break; case 6: std::swap(x, y); break; case 7: x ^= y; x %= M; break; default: y = (y + x + 1) % M; }
          nx[s][a] = y * M + x;
        }
      }
      for (int s = 0; s < n; s++) goals.push_back(s);
    } else if (k < 18) {  // three-dial combination lock (base 10), goal combination drawn per episode; k=17 couples neighbouring dials
      name = k == 16 ? "Lock" : "Lock-coupled"; n = 1000; digits_base(10, 3); ng = 3; GB = 10; nx.assign(n, {});
      for (int s = 0; s < n; s++) for (int a = 0; a < 9; a++) {
        int d[3] = {dg[s][0], dg[s][1], dg[s][2]};
        if (a < 6) { const int i = a / 2, sg = a % 2 ? -1 : 1; d[i] = (d[i] + sg + 10) % 10; if (k == 17) d[(i + 1) % 3] = (d[(i + 1) % 3] + 1) % 10; }
        else if (a < 9 && k == 16) { const int i = a - 6; d[i] = (d[i] + 5) % 10; }
        else { const int t = d[0]; d[0] = d[1]; d[1] = d[2]; d[2] = t; }
        nx[s][a] = d[0] + 10 * d[1] + 100 * d[2];
      }
      for (int s = 0; s < n; s++) goals.push_back(s);
    } else {  // 64-state graphs: k=18 ring with random step sizes, k=19 an independent random permutation per action
      name = k == 18 ? "Ring64" : "Perm64"; n = 64; digits_base(2, 6); ng = 1; gsingle = true; GB = n; nx.assign(n, {});
      for (int a = 0; a < 9; a++) {
        std::vector<int> p(n); for (int i = 0; i < n; i++) p[i] = i;
        if (k == 19) for (int i = n - 1; i > 0; i--) std::swap(p[i], p[rnd(i + 1)]);
        const int st = (rnd(2) ? 1 : -1) * (1 + rnd(9));
        for (int s = 0; s < n; s++) nx[s][a] = k == 18 ? ((s + st) % n + n) % n : p[s];
      }
      for (int s = 0; s < n; s++) goals.push_back(s);
    }
  }
  Vec feat(int s, int g, int tick) const {
    Vec o{}; auto sc = [](int v, int b) { return b <= 1 ? 0.f : 2.f * (float)v / (float)(b - 1) - 1.f; };
    for (int i = 0; i < nd; i++) o[i] = sc(dg[s][i], B);
    if (ng > 0) { if (gsingle) o[nd] = sc(g, GB); else for (int i = 0; i < ng; i++) o[nd + i] = sc(dg[g][i], GB); }
    o[9] = clampf((float)tick / 25.f - 1.f, -1, 1);
    return o;
  }
  bool random_goal() const { return ng > 0; }
};

// ---- External environment (scenario id 3200 = game 100): any simulator reachable through a line protocol on a child process's stdin/stdout, e.g.
// nexus_gym_bridge.py for Gymnasium (see that file). One shared process; episodes run one at a time. Reward: -0.01 per step, +4 at a goal, -1 on failure.
struct ExtProc {
  pid_t pid = -1; FILE* to = nullptr; FILE* from = nullptr; int dim = 0, max_steps = 200;
  bool start(const std::string& cmd) {
    int a[2], b[2];
    if (pipe(a) != 0 || pipe(b) != 0) return false;
    fcntl(a[1], F_SETFD, FD_CLOEXEC); fcntl(b[0], F_SETFD, FD_CLOEXEC);
    pid = fork();
    if (pid == 0) { dup2(a[0], 0); dup2(b[1], 1); execl("/bin/sh", "sh", "-c", cmd.c_str(), (char*)nullptr); _exit(127); }
    close(a[0]); close(b[1]);
    if (pid < 0) return false;
    to = fdopen(a[1], "w"); from = fdopen(b[0], "r");
    std::string r;
    if (!ask("", r, true)) return false;
    std::istringstream is(r); std::string w; is >> w >> dim >> max_steps;
    return w == "ready" && dim >= 1 && dim <= OBS && max_steps >= 1;
  }
  bool ask(const std::string& line, std::string& reply, bool no_send = false) {
    if (!to || !from) return false;
    if (!no_send) { std::fprintf(to, "%s\n", line.c_str()); std::fflush(to); }
    char buf[8192];
    if (!std::fgets(buf, sizeof buf, from)) return false;
    reply = buf; return true;
  }
  ~ExtProc() { if (to) { std::fprintf(to, "quit\n"); std::fflush(to); std::fclose(to); } if (from) std::fclose(from); if (pid > 0) waitpid(pid, nullptr, 0); }
};
inline std::string g_ext_cmd;
inline std::shared_ptr<ExtProc> g_ext;
inline ExtProc& ext_proc() {
  if (!g_ext) {
    if (g_ext_cmd.empty()) throw std::runtime_error("scenario 3200 needs --ext \"command\"");
    auto e = std::make_shared<ExtProc>();
    if (!e->start(g_ext_cmd)) throw std::runtime_error("cannot start the external environment: " + g_ext_cmd);
    g_ext = e;
  }
  return *g_ext;
}

struct World {
  std::mt19937 rng;
  float x = 0, y = 0, vx = 0, vy = 0, gx = 0, gy = 0, ox = 0, oy = 0, windx = 0, windy = 0;
  Dyn d;
  int tick = 0, lab = -1;
  float sense = 0.f, kx = 0, ky = 0;
  bool keydoor = false, has_key = false;
  bool invert = false, gmove = false, chain = false;
  int stage = 0;
  float k2x = 0, k2y = 0, gvx = 0, gvy = 0;
  bool sparse = cfg.sparse, label = cfg.label, force_dense = false;
  int mv = 1, mt = 0;  // game 2 (NumberChain): current number and target, both in 0..255
  float pth = 0, pw = 0;  // game 3 (Pendulum swing-up): angle from upright and angular velocity
  int sw = 0;             // game 4 (Switches): bit mask of eight lights, 1 = on
  int maxT() const { return game == 100 ? (ext_over ? 0 : ext_proc().max_steps) : game >= 10 ? HELD_STEPS : game == 1 || game == 3 ? GAME_STEPS : game == 2 || game == 5 ? 20 : game == 4 ? 40 : MAX_STEPS; }
  static float wrap_pi(float a) { while (a > 3.14159265f) a -= 6.2831853f; while (a < -3.14159265f) a += 6.2831853f; return a; }
  int game = 0;  // 1: MountainCar-v0 (state in mcx, mcv); the reward is the game's own: -0.01 per step (a rescaled -1) and +4 on reaching the flag
  float mcx = 0, mcv = 0;
  explicit World(uint32_t seed = 42) : rng(seed) { reset(scenario(0)); }

  Puzzle pz; int pz_s = 0, pz_g = 0;  // held-out family (game 10..29)
  Vec eo{}; bool ext_over = false;    // external environment (game 100): last observation, episode finished (failure / step limit)
  Vec obs() const {
    if (game == 100) return eo;
    if (game >= 10) return pz.feat(pz_s, pz_g, tick);
    if (game == 3) return Vec{std::cos(pth), std::sin(pth), clampf(pw / 8.f, -1, 1), 0, 0, 0, 0, 0, 0, 0};
    if (game == 4) { Vec o{}; for (int i = 0; i < 8; i++) o[i] = (sw >> i & 1) ? 1.f : -1.f; o[8] = clampf((float)tick / 20.f - 1.f, -1, 1); return o; }
    if (game == 5) return Vec{clampf((float)mv / 48.f - 1.f, -1, 1), clampf((float)tick / 10.f - 1.f, -1, 1), clampf((float)mt / 48.f - 1.f, -1, 1), 0, clampf((float)(mt - mv) / 96.f, -1, 1), 0, 0, 0, 0, 0};
    if (game == 2) return Vec{clampf((float)mv / 128.f - 1.f, -1, 1), clampf((float)tick / 10.f - 1.f, -1, 1), clampf((float)mt / 128.f - 1.f, -1, 1), 0, clampf((float)(mt - mv) / 128.f, -1, 1), 0, 0, 0, 0, 0};
    if (game == 1) return Vec{clampf((mcx + .3f) / .9f, -1, 1), 0.f, clampf(mcv / .07f, -1, 1), 0, 0, 0, 0, 0, 0, 0};  // only what the game shows: position and velocity
    float flag = (label && lab >= 0) ? (lab ? 1.f : -1.f) : 0.f;
    Vec o{clampf(x, -1, 1), clampf(y, -1, 1),
          clampf(vx, -.3f, .3f) / .3f, clampf(vy, -.3f, .3f) / .3f,
          clampf(gx - x, -2, 2) / 2, clampf(gy - y, -2, 2) / 2,
          clampf(ox - x, -2, 2) / 2, clampf(oy - y, -2, 2) / 2,
          flag, clampf((std::abs(windx) + std::abs(windy)) / .28f, 0, 1)};
    if (cfg.show_phase) o[9] = 2.f * (float)(tick % (2 * d.flip)) / (float)(2 * d.flip) - 1.f;  // sawtooth over the full wind cycle: time to the next flip and its sign
    else if (cfg.show_wind) o[9] = clampf((windx + windy) / .28f, -1.f, 1.f);  // diagnostic: the SIGNED wind becomes observable
    if (sense > 0.f) {  // hidden-goal task: the goal vector is only visible inside the sense radius
      bool vis = std::hypot(gx - x, gy - y) <= sense;
      if (!vis) { o[4] = 0.f; o[5] = 0.f; }
      o[8] = vis ? 1.f : -1.f;
    }
    if (keydoor) {  // slots 6/7 carry the vector to the key (zero once collected); slot 8 the phase flag
      o[6] = has_key ? 0.f : clampf(kx - x, -2, 2) / 2;
      o[7] = has_key ? 0.f : clampf(ky - y, -2, 2) / 2;
      o[8] = has_key ? 1.f : -1.f;
    }
    if (chain) {  // slots 6/7: vector to the next key (zero when both are collected); slot 8: progress -1 / 0 / +1
      const float tx = stage == 0 ? kx : k2x, ty = stage == 0 ? ky : k2y;
      o[6] = stage >= 2 ? 0.f : clampf(tx - x, -2, 2) / 2;
      o[7] = stage >= 2 ? 0.f : clampf(ty - y, -2, 2) / 2;
      o[8] = (float)stage - 1.f;
    }
    return o;
  }
  void place_obstacle(MapMode m) {
    std::uniform_real_distribution<float> u(-.82f, .82f), u01(0.f, 1.f);
    if (m != MapMode::Random) {
      for (int tries = 0; tries < 200; tries++) {
        if (m == MapMode::Blocker) {
          float t = .35f + .3f * u01(rng), dx = gx - x, dy = gy - y, L = std::max(1e-3f, std::hypot(dx, dy));
          float j = (u01(rng) - .5f) * .1f;
          ox = x + dx * t - dy / L * j; oy = y + dy * t + dx / L * j;
        } else {
          float th = u01(rng) * 6.2831853f;
          ox = gx + .28f * std::cos(th); oy = gy + .28f * std::sin(th);
        }
        if (std::abs(ox) <= .9f && std::abs(oy) <= .9f && std::hypot(ox - x, oy - y) >= .3f && std::hypot(ox - gx, oy - gy) >= .22f) return;
      }
    }
    do { ox = u(rng) * .6f; oy = u(rng) * .6f; }
    while (std::hypot(ox - gx, oy - gy) < .35f || std::hypot(ox - x, oy - y) < .3f);
  }
  Vec reset(const Scn& s) {
    std::uniform_real_distribution<float> u(-.82f, .82f);
    game = s.game; tick = 0;
    if (game == 100) {
      keydoor = chain = gmove = invert = false; sense = 0.f; has_key = false; stage = 0; ext_over = false;
      std::string r; ExtProc& e = ext_proc();
      if (!e.ask("reset", r)) throw std::runtime_error("external environment stopped answering");
      std::istringstream is(r); std::string w; is >> w; eo = Vec{}; for (int i = 0; i < e.dim; i++) is >> eo[i];
      return eo;
    }
    if (game >= 10) {
      keydoor = chain = gmove = invert = false; sense = 0.f; has_key = false; stage = 0;
      if (pz.kind != game - 10) pz.build(game - 10);
      for (int tries = 0; tries < 200; tries++) {
        if (pz.ng > 0) { pz_s = (int)(rng() % (uint32_t)pz.n); pz_g = pz.goals[rng() % pz.goals.size()]; }
        else { pz_g = pz.goal; pz_s = pz.goals[rng() % pz.goals.size()]; }
        int d = -1; Puzzle::dist_first(pz.nx, pz_s, pz_g, &d);
        if (pz_s != pz_g && d >= 2 && d <= HELD_STEPS - 15) break;
      }
      return obs();
    }
    if (game >= 3) {
      keydoor = chain = gmove = invert = false; sense = 0.f; has_key = false; stage = 0;
      if (game == 3) { std::uniform_real_distribution<float> a(-3.14159f, 3.14159f), b(-1.f, 1.f); pth = a(rng); pw = b(rng); }
      if (game == 4) { do { sw = 255; const int n = 6 + (int)(rng() % 7u); for (int i = 0; i < n; i++) { const int pr = (int)(rng() % 8u); for (int k = std::max(0, pr - 1); k <= std::min(7, pr + 1); k++) sw ^= 1 << k; } } while (sw == 255); }
      if (game == 5) { mv = 1; mt = 2 + (int)(rng() % 95u); }
      return obs();
    }
    if (game == 2) { mv = 1; mt = 20 + (int)(rng() % 181u); keydoor = chain = gmove = invert = false; sense = 0.f; has_key = false; stage = 0; return obs(); }
    if (game == 1) { std::uniform_real_distribution<float> st(-.6f, -.4f); mcx = st(rng); mcv = 0; keydoor = chain = gmove = invert = false; sense = 0.f; has_key = false; stage = 0; return obs(); }
    d = s.d; lab = s.lab; sense = s.sense;
    x = u(rng); y = u(rng); vx = vy = 0;
    do { gx = u(rng); gy = u(rng); } while (std::hypot(gx - x, gy - y) < .6f);
    place_obstacle(s.m);
    keydoor = s.keydoor; has_key = false;
    if (keydoor) do { kx = u(rng); ky = u(rng); } while (std::hypot(kx - x, ky - y) < .5f || std::hypot(kx - gx, ky - gy) < .5f);
    invert = s.invert; gmove = s.gmove; chain = s.chain; stage = 0;
    if (chain) {
      do { kx = u(rng); ky = u(rng); } while (std::hypot(kx - x, ky - y) < .5f || std::hypot(kx - gx, ky - gy) < .5f);
      do { k2x = u(rng); k2y = u(rng); } while (std::hypot(k2x - x, k2y - y) < .5f || std::hypot(k2x - gx, k2y - gy) < .5f || std::hypot(k2x - kx, k2y - ky) < .5f);
    }
    if (gmove) { std::uniform_real_distribution<float> th(0.f, 6.2831853f); float a = th(rng); gvx = .035f * std::cos(a); gvy = .035f * std::sin(a); }
    tick = 0;
    windx = d.wx; windy = d.wy;
    return obs();
  }
  Transition step(int a) {
    Transition tr;
    tr.s = obs(); tr.a = a; tr.task = (uint8_t)std::max(lab, 0);
    if (game == 100) {
      std::string r; ExtProc& e = ext_proc();
      if (!e.ask("step " + std::to_string(a % 9), r)) throw std::runtime_error("external environment stopped answering");
      std::istringstream is(r); std::string w; is >> w; Vec n{}; for (int i = 0; i < e.dim; i++) is >> n[i];
      int fail = 0, goal = 0, trunc = 0; is >> fail >> goal >> trunc;
      tick++; eo = n;
      tr.r = -.01f + (goal ? 4.f : 0.f) - (fail ? 1.f : 0.f); tr.ns = eo; tr.done = goal != 0;
      if (fail) tr.term = true;
      if (fail || trunc || goal) ext_over = true;
      return tr;
    }
    if (game >= 10) {
      pz_s = pz.nx[pz_s][a % 9]; tick++;
      const bool done = pz_s == pz_g;
      tr.r = -.01f + (done ? 4.f : 0.f); tr.ns = obs(); tr.done = done;
      return tr;
    }
    if (game == 3) {  // Pendulum swing-up (Gym dynamics: g = 10, l = m = 1, dt = .05, torque -2 / 0 / +2 from the horizontal part of the action); goal: upright and slow
      static const int dxs[8] = {1, 1, 0, -1, -1, -1, 0, 1};
      const float u = 2.f * (float)(a > 0 ? dxs[a - 1] : 0);
      pw = clampf(pw + (15.f * std::sin(pth) + 3.f * u) * .05f, -8.f, 8.f);
      pth = wrap_pi(pth + pw * .05f);
      tick++;
      const bool done = std::fabs(pth) < .15f && std::fabs(pw) < 1.f;
      tr.r = -.01f + (done ? 4.f : 0.f); tr.ns = obs(); tr.done = done;
      return tr;
    }
    if (game == 4) {  // Switches: pressing button p (a mod 8) flips lights p-1, p, p+1; goal: all eight on
      const int pr = a % 8;
      for (int k = std::max(0, pr - 1); k <= std::min(7, pr + 1); k++) sw ^= 1 << k;
      tick++;
      const bool done = sw == 255;
      tr.r = -.01f + (done ? 4.f : 0.f); tr.ns = obs(); tr.done = done;
      return tr;
    }
    if (game == 5) {  // ModChain: reach the target modulo 97 with +1 +5 +17 x2 x3 -1 -7 x5 2x+1
      switch (a) { case 0: mv = (mv + 1) % 97; break; case 1: mv = (mv + 5) % 97; break; case 2: mv = (mv + 17) % 97; break; case 3: mv = (mv * 2) % 97; break;
                   case 4: mv = (mv * 3) % 97; break; case 5: mv = (mv + 96) % 97; break; case 6: mv = (mv + 90) % 97; break; case 7: mv = (mv * 5) % 97; break; default: mv = (mv * 2 + 1) % 97; }
      tick++;
      const bool done = mv == mt;
      tr.r = -.01f + (done ? 4.f : 0.f); tr.ns = obs(); tr.done = done;
      return tr;
    }
    if (game == 2) {  // NumberChain: start at 1, reach the target with +1 +2 +4 +8 x2 x3 -1 -4 /2 (an invalid result leaves the number unchanged); 20 steps
      int nv = mv;
      switch (a) { case 0: nv = mv + 1; break; case 1: nv = mv + 2; break; case 2: nv = mv + 4; break; case 3: nv = mv + 8; break;
                   case 4: nv = mv * 2; break; case 5: nv = mv * 3; break; case 6: nv = mv - 1; break; case 7: nv = mv - 4; break; default: nv = (mv % 2 == 0) ? mv / 2 : mv; }
      if (nv >= 0 && nv <= 255) mv = nv;
      tick++;
      const bool done = mv == mt;
      tr.r = -.01f + (done ? 4.f : 0.f);
      tr.ns = obs(); tr.done = done;
      return tr;
    }
    if (game == 1) {  // MountainCar-v0 physics exactly as in Gym: force .001, gravity .0025, speed limit .07, walls at -1.2 / .6, flag at .5
      static const int dxs[8] = {1, 1, 0, -1, -1, -1, 0, 1};
      const int push = a > 0 ? dxs[a - 1] : 0;  // the 9 arena actions are reduced to left / none / right by their horizontal part
      mcv += (float)push * .001f + std::cos(3.f * mcx) * -.0025f;
      mcv = clampf(mcv, -.07f, .07f);
      mcx += mcv;
      mcx = clampf(mcx, -1.2f, .6f);
      if (mcx <= -1.2f && mcv < 0) mcv = 0;
      tick++;
      const bool done = mcx >= .5f && mcv >= 0;
      tr.r = -.01f + (done ? 4.f : 0.f);
      tr.ns = obs(); tr.done = done;
      return tr;
    }
    float ax = 0, ay = 0;
    if (a > 0) {
      static const int dx[8] = {1, 1, 0, -1, -1, -1, 0, 1}, dy[8] = {0, 1, 1, 1, 0, -1, -1, -1};
      ax = (float)dx[a - 1]; ay = (float)dy[a - 1];
      if (invert && has_key) { ax = -ax; ay = -ay; }  // the rules changed: controls are mirrored once the key is collected
    }
    vx = clampf(d.f * vx + d.sp * ax + windx, -.27f, .27f);
    vy = clampf(d.f * vy + d.sp * ay + windy, -.27f, .27f);
    x += vx; y += vy;
    if (x < -.95f || x > .95f) { x = clampf(x, -.95f, .95f); vx *= -.55f; }
    if (y < -.95f || y > .95f) { y = clampf(y, -.95f, .95f); vy *= -.55f; }
    tick++;
    if (tick % d.flip == 0) { windx = -windx; windy = -windy; }
    if (gmove) {  // the goal drifts and bounces inside the arena
      gx += gvx; gy += gvy;
      if (gx < -.82f || gx > .82f) { gx = clampf(gx, -.82f, .82f); gvx = -gvx; }
      if (gy < -.82f || gy > .82f) { gy = clampf(gy, -.82f, .82f); gvy = -gvy; }
    }
    const bool sp = (sparse || sense > 0.f) && !force_dense;
    if (chain) {  // collect key 1, then key 2, then reach the goal
      bool ev = false;
      if (stage < 2) {
        const float tx = stage == 0 ? kx : k2x, ty = stage == 0 ? ky : k2y;
        if (std::hypot(tx - x, ty - y) < .09f) { stage++; ev = true; }
      }
      const float dg = std::hypot(gx - x, gy - y);
      const bool done = stage == 2 && dg < .085f;
      const float td = stage == 0 ? std::hypot(kx - x, ky - y) : stage == 1 ? std::hypot(k2x - x, k2y - y) : dg;
      tr.r = (sp ? -.01f : -.035f - .55f * td) + (ev ? 1.f : 0.f) + (done ? 4.f : 0.f);
      tr.ns = obs(); tr.done = done;
      return tr;
    }
    if (keydoor) {  // the goal is inert until the key was collected; the obstacle plays no role here
      bool key_event = false;
      if (!has_key && std::hypot(kx - x, ky - y) < .09f) { has_key = true; key_event = true; }
      float dg = std::hypot(gx - x, gy - y);
      bool done = has_key && dg < .085f;
      float td = has_key ? dg : std::hypot(kx - x, ky - y);
      tr.r = (sp ? -.01f : -.035f - .55f * td) + (key_event ? 1.f : 0.f) + (done ? 4.f : 0.f);
      tr.ns = obs(); tr.done = done;
      return tr;
    }
    Reward rw = reward_fn(std::hypot(gx - x, gy - y), std::hypot(ox - x, oy - y), sp);
    tr.ns = obs(); tr.done = rw.done; tr.r = rw.r;
    return tr;
  }
};

// ---------------------------------------------------------------------------------------------
// Agent: ensemble world model + CEM planner (MPC) + optional RND novelty + optional double-Q value.
// ---------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------
// v9 / item 1: discrete symbols. Every active observation dimension gets a scalar codebook that is refined online (k-means, count-based
// step); a state is then a tuple of tokens. A categorical next-token model (embeddings -> MLP -> one softmax per dimension, cross-entropy)
// replaces the regression of deltas, and the planner SNAPS every predicted step to the nearest symbol, so prediction errors do not drift.
// ---------------------------------------------------------------------------------------------
struct Lin {  // dense layer with run-time sizes and its own Adam state
  int in = 0, out = 0;
  uint64_t t = 0;
  std::vector<float> w, b, mw, vw, mb, vb, gw, gb;
  void init(int i, int o, uint32_t seed, float gain = 1.f) {
    in = i; out = o; t = 0;
    w.assign((size_t)i * o, 0.f); b.assign((size_t)o, 0.f);
    mw = vw = gw = w; mb = vb = gb = b;
    std::mt19937 r(seed);
    std::normal_distribution<float> d(0.f, std::sqrt(gain / (float)i));
    for (auto& x : w) x = d(r);
  }
  void forward(const float* x, float* y) const {
    for (int o = 0; o < out; o++) {
      const float* wr = &w[(size_t)o * in];
      float a = b[o];
      for (int i = 0; i < in; i++) a += wr[i] * x[i];
      y[o] = a;
    }
  }
  void backward(const float* x, const float* dy, float* dx) {  // accumulate gradients; add dL/dx to dx (may be null)
    for (int o = 0; o < out; o++) {
      const float g = dy[o];
      if (g == 0.f) continue;
      gb[o] += g;
      float* gr = &gw[(size_t)o * in];
      const float* wr = &w[(size_t)o * in];
      for (int i = 0; i < in; i++) { gr[i] += g * x[i]; if (dx) dx[i] += g * wr[i]; }
    }
  }
  void step(float lr, float scale) {
    t++;
    const float c1 = 1.f - std::pow(.9f, (float)t), c2 = 1.f - std::pow(.999f, (float)t);
    auto upd = [&](std::vector<float>& p, std::vector<float>& m, std::vector<float>& v, std::vector<float>& g) {
      for (size_t i = 0; i < p.size(); i++) {
        const float gg = g[i] * scale;
        m[i] = .9f * m[i] + .1f * gg; v[i] = .999f * v[i] + .001f * gg * gg;
        p[i] -= lr * (m[i] / c1) / (std::sqrt(v[i] / c2) + 1e-8f);
        g[i] = 0.f;
      }
    };
    upd(w, mw, vw, gw); upd(b, mb, vb, gb);
  }
  static void putv(std::ostream& o, const std::vector<float>& v) { uint64_t n = v.size(); wr(o, n); o.write(reinterpret_cast<const char*>(v.data()), (std::streamsize)(n * sizeof(float))); }
  static bool getv(std::istream& i, std::vector<float>& v, uint64_t expect) {
    uint64_t n = 0; rd(i, n);
    if (!i || n != expect) return false;
    v.assign((size_t)n, 0.f);
    i.read(reinterpret_cast<char*>(v.data()), (std::streamsize)(n * sizeof(float)));
    return (bool)i;
  }
  void save(std::ostream& o) const { wr(o, in); wr(o, out); wr(o, t); putv(o, w); putv(o, b); putv(o, mw); putv(o, vw); putv(o, mb); putv(o, vb); }
  bool load(std::istream& i) {
    int a = 0, c = 0; uint64_t tt = 0;
    rd(i, a); rd(i, c); rd(i, tt);
    if (!i || a != in || c != out) return false;
    std::vector<float> W, B, MW, VW, MB, VB;
    if (!getv(i, W, (uint64_t)in * out) || !getv(i, B, out) || !getv(i, MW, (uint64_t)in * out) || !getv(i, VW, (uint64_t)in * out) || !getv(i, MB, out) || !getv(i, VB, out)) return false;
    for (float x : W) if (!std::isfinite(x)) return false;
    w = W; b = B; mw = MW; vw = VW; mb = MB; vb = VB; t = tt;
    std::fill(gw.begin(), gw.end(), 0.f); std::fill(gb.begin(), gb.end(), 0.f);
    return true;
  }
};

// ---- v10: symbolic rule synthesis (--synth) and externally supplied rule hypotheses (--rules FILE) ----
// For every (action, observation dimension) the agent looks for a short program  ns_j = alpha*s_j + beta  that is consistent with ALL its
// experience, where a result that leaves the legal range or the integer lattice means "unchanged" (a guard). Candidates are a small fixed
// alpha set plus any hypotheses from the rules file (written by a person or a language model); beta is fitted. A rule is only used if it
// explains >= 99.5% of >= 30 samples and, when it is not the identity, >= 5 changing samples. Everything else stays with the neural model.
struct Synth {
  // type 5: the value dimension k takes after this step according to its own verified rule (only used inside rotation rules)
  struct Term { int type = 0; int k = 0, m = 0; float w = 1.f, c = 0.f; };  // 0 linear s_k, 1 square, 2 product s_k*s_m, 3 sin(w s_k), 4 cos(w s_k)
  struct Rule { bool ok = false; float alpha = 1.f, beta = 0.f, lo = -1.f, hi = 1.f, q = 0.f; bool lib = false, absolute = false; float b0 = 0.f, acc = 0.f; std::vector<Term> terms;
                int rot = 0, rp = -1;  // rotation rule: this dim is the cos (rot=1) or sin (rot=2) of an angle whose partner dim is rp; terms/b0 give the angle INCREMENT
                bool has_cond = false; std::vector<Term> ct; float cb = 0.f, cthr = 0.f, cval = 0.f; int cdir = 0; };  // optional reset: if cb + sum(ct) < / > cthr then the result is cval
  std::array<std::array<Rule, OBS>, ACT> r{};
  std::vector<std::array<float, 4>> hyp;  // action, dim, alpha, beta
  struct HypF { int a = 0, j = 0; float b0 = 0.f; std::vector<Term> terms; std::string text; bool has_cond = false; std::vector<Term> ct; float cb = 0.f, cthr = 0.f, cval = 0.f; int cdir = 0; };
  std::vector<HypF> hypf;                  // formula hypotheses:  f ACTION DIM = c + c1*s2 + c2*sin(w*s0) + c3*cos(w*s0) + c4*s1*s2 + c5*s3^2 + c6*s1
  std::vector<std::string> verdicts;
  static bool parse_formula(const std::string& expr, HypF& h) {
    std::string e; for (char c : expr) if (!std::isspace((unsigned char)c)) e += c;
    const std::regex num("^[-+]?[0-9]*\\.?[0-9]+(?:[eE][-+]?[0-9]+)?$"), coef("^([-+]?[0-9]*\\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\\*(.+)$");
    const std::regex lin("^s(\\d+)$"), sq("^s(\\d+)\\^2$"), prod("^s(\\d+)\\*s(\\d+)$"), trig("^(sin|cos)\\(([-+]?[0-9]*\\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\\*s(\\d+)\\)$");
    std::vector<std::string> parts; std::string cur; int depth = 0;
    for (size_t i = 0; i < e.size(); i++) {
      if (e[i] == '(') depth++;
      if (e[i] == ')') depth--;
      if (e[i] == '+' && depth == 0 && !(i > 0 && (e[i - 1] == 'e' || e[i - 1] == 'E') && i > 1 && std::isdigit((unsigned char)e[i - 2]))) { if (!cur.empty()) parts.push_back(cur); cur.clear(); } else cur += e[i];
    }
    if (!cur.empty()) parts.push_back(cur);
    for (const auto& t0 : parts) {
      std::smatch m; std::string t = t0; Term T; float c = 1.f;
      if (std::regex_match(t, num)) { h.b0 += std::stof(t); continue; }
      if (std::regex_match(t, m, coef)) { c = std::stof(m[1]); t = m[2]; if (std::regex_match(t, num)) { h.b0 += c * std::stof(t); continue; } }
      if (std::regex_match(t, m, lin)) { T.type = 0; T.k = std::stoi(m[1]); }
      else if (std::regex_match(t, m, sq)) { T.type = 1; T.k = std::stoi(m[1]); }
      else if (std::regex_match(t, m, prod)) { T.type = 2; T.k = std::stoi(m[1]); T.m = std::stoi(m[2]); }
      else if (std::regex_match(t, m, trig)) { T.type = m[1] == "sin" ? 3 : 4; T.w = std::stof(m[2]); T.k = std::stoi(m[3]); }
      else return false;
      if (T.k < 0 || T.k >= OBS || T.m < 0 || T.m >= OBS) return false;
      T.c = c; h.terms.push_back(T);
    }
    return !parts.empty();
  }
  int nrules = 0, hyp_ok = 0, hyp_bad = 0, nlib = 0;
  bool ready = false;
  static constexpr float TOL = 2e-3f;
  static float lattice_ok(float pred, float s, float q) {
    if (q <= 0.f) return true;
    const float k = (pred - s) / q;
    return std::fabs(k - std::round(k)) < .05f;
  }
  static bool legal(const Rule& u, float s, float pred) { return pred >= u.lo - 1e-6f && pred <= u.hi + 1e-6f && lattice_ok(pred, s, u.q); }
  float apply1(const Rule& u, float s) const {
    const float p = u.alpha * s + u.beta;
    return legal(u, s, p) ? clampf(p, -1.f, 1.f) : s;
  }
  bool load_rules(const std::string& path) {
    std::ifstream f(path);
    if (!f) return false;
    std::string line;
    while (std::getline(f, line)) {
      const size_t h = line.find('#');
      if (h != std::string::npos) line.resize(h);
      std::istringstream is(line);
      std::string first; is >> first;
      if (first == "f") {  // formula hypothesis
        HypF h; char eq = 0; is >> h.a >> h.j >> eq;
        std::string rest; std::getline(is, rest);
        h.text = line;
        bool cond_ok = true;
        const size_t wp = rest.find("| when");
        if (wp != std::string::npos) {  // reset clause:  | when <formula> < 3.5 -> 0.0   (or  > )
          std::string cl = rest.substr(wp + 6); rest.resize(wp);
          const size_t arrow = cl.find("->"), lt = cl.find('<'), gt = cl.find('>');
          const size_t op = lt != std::string::npos && lt < arrow ? lt : gt;
          cond_ok = arrow != std::string::npos && op != std::string::npos && op < arrow;
          if (cond_ok) {
            HypF c; cond_ok = parse_formula(cl.substr(0, op), c);
            if (cond_ok) { try { h.cthr = std::stof(cl.substr(op + 1, arrow - op - 1)); h.cval = std::stof(cl.substr(arrow + 2)); } catch (...) { cond_ok = false; } }
            h.has_cond = cond_ok; h.ct = c.terms; h.cb = c.b0; h.cdir = op == lt ? -1 : 1;
          }
        }
        if (cond_ok && eq == '=' && h.a >= 0 && h.a < ACT && h.j >= 0 && h.j < OBS && parse_formula(rest, h)) hypf.push_back(h);
        else verdicts.push_back("could not parse: " + line);
        continue;
      }
      std::istringstream is2(line);
      float a, d, al, be;
      if (is2 >> a >> d >> al >> be) hyp.push_back({a, d, al, be});
    }
    return true;
  }
  void fit(const std::vector<Transition>& rep) {
    nrules = 0; hyp_ok = 0; hyp_bad = 0; nlib = 0;
    verdicts.erase(std::remove_if(verdicts.begin(), verdicts.end(), [](const std::string& v) { return v.rfind("could not parse", 0) != 0; }), verdicts.end());
    std::vector<int> vdims;  // dimensions that vary in the data
    for (int j = 0; j < OBS; j++) { float lo = 1e9f, hi = -1e9f; for (const auto& t : rep) { lo = std::min(lo, t.s[j]); hi = std::max(hi, t.s[j]); if (hi - lo > 1e-4f) break; } if (hi - lo > 1e-4f) vdims.push_back(j); }
    for (auto& row : r) for (auto& u : row) u = Rule{};
    std::vector<const Transition*> by[ACT];
    for (const auto& t : rep) by[t.a].push_back(&t);
    std::vector<char> hit(hyp.size(), 0);
    for (int a = 0; a < ACT; a++) {
      if (by[a].size() < 30) continue;
      for (int j = 0; j < OBS; j++) {
        Rule base;
        float q = 1e9f, lo = 1e9f, hi = -1e9f;
        int changed = 0;
        for (const auto* t : by[a]) {
          const float d = std::fabs(t->ns[j] - t->s[j]);
          if (d > 1e-5f) { changed++; q = std::min(q, d); }
          lo = std::min(lo, t->ns[j]); hi = std::max(hi, t->ns[j]);
        }
        base.lo = lo; base.hi = hi; base.q = changed ? q : 0.f;
        if (changed == 0) { base.ok = true; r[a][j] = base; nrules++; continue; }  // never changes: identity
        std::vector<std::array<float, 2>> cand;  // alpha, beta
        const float alphas[] = {0.f, .5f, 1.f, 2.f, 3.f, -1.f, -2.f};
        std::vector<float> diffs;
        for (float al : alphas) {
          diffs.clear();
          for (const auto* t : by[a]) if (std::fabs(t->ns[j] - t->s[j]) > 1e-5f) diffs.push_back(t->ns[j] - al * t->s[j]);
          std::nth_element(diffs.begin(), diffs.begin() + diffs.size() / 2, diffs.end());
          cand.push_back({al, diffs[diffs.size() / 2]});
        }
        const size_t nstd = cand.size();
        std::vector<int> hid;
        for (size_t h = 0; h < hyp.size(); h++) if ((int)hyp[h][0] == a && (int)hyp[h][1] == j) { cand.push_back({hyp[h][2], hyp[h][3]}); hid.push_back((int)h); }
        double bestf = 0; int bestc = -1, bestch = 0;
        for (size_t c = 0; c < cand.size(); c++) {
          Rule u = base; u.alpha = cand[c][0]; u.beta = cand[c][1];
          int good = 0, chg = 0;
          for (const auto* t : by[a]) {
            const float p = u.alpha * t->s[j] + u.beta;
            const bool use = legal(u, t->s[j], p);
            const float pr = use ? clampf(p, -1.f, 1.f) : t->s[j];
            if (std::fabs(pr - t->ns[j]) < TOL) { good++; if (std::fabs(t->ns[j] - t->s[j]) > 1e-5f) chg++; }
          }
          const double f = (double)good / by[a].size();
          if (f > bestf + 1e-9) { bestf = f; bestc = (int)c; bestch = chg; }
          if (c >= nstd && f >= .995 && chg >= 5) hit[hid[c - nstd]] = 1;
        }
        if (bestc >= 0 && bestf >= .995 && bestch >= 5) {
          base.alpha = cand[bestc][0]; base.beta = cand[bestc][1]; base.ok = true;
          r[a][j] = base; nrules++;
        } else if (cfg.synth_lib) {
          Rule u;
          if (fit_lib(by[a], j, vdims, u)) { r[a][j] = u; nrules++; nlib++; } else r[a][j].acc = u.acc;
        }
      }
    }
    if (cfg.synth_lib) {  // rotation rules for (cos, sin) pairs that no affine/library rule explained
      for (int a = 0; a < ACT; a++) {
        if (by[a].size() < 200) continue;
        for (size_t x = 0; x < vdims.size(); x++) for (size_t y2 = x + 1; y2 < vdims.size(); y2++) {
          const int j1 = vdims[x], j2 = vdims[y2];
          if (r[a][j1].ok || r[a][j2].ok) continue;
          Rule u1, u2;
          if (fit_rot(a, by[a], j1, j2, vdims, u1, u2)) { r[a][j1] = u1; r[a][j2] = u2; nrules += 2; nlib += 2; }
        }
      }
    }
    for (size_t h = 0; h < hyp.size(); h++) (hit[h] ? hyp_ok : hyp_bad)++;
    for (const auto& h : hypf) {  // formula hypotheses are tested on ALL experience with that action
      if (by[h.a].size() < 30) { hyp_bad++; verdicts.push_back("unsupported (fewer than 30 samples of this action): " + h.text); continue; }
      Rule u; u.lib = true; u.absolute = true; u.b0 = h.b0; u.terms = h.terms;
      u.has_cond = h.has_cond; u.ct = h.ct; u.cb = h.cb; u.cthr = h.cthr; u.cval = h.cval; u.cdir = h.cdir;
      float lo = 1e9f, hi = -1e9f;
      for (const auto* t : by[h.a]) { lo = std::min(lo, t->ns[h.j]); hi = std::max(hi, t->ns[h.j]); }
      u.lo = -1.f; u.hi = 1.f;
      int good = 0;
      std::ostringstream miss; int nm = 0;
      for (const auto* t : by[h.a]) {
        const float pr = eval_lib(u, t->s, h.j);
        if (std::fabs(pr - t->ns[h.j]) < TOL) good++;
        else if (nm < 5 && (&t - &by[h.a][0]) % 7 == 0) { nm++; miss << " [s0=" << std::setprecision(3) << t->s[0] << " s2=" << t->s[2] << " predicted " << pr << " actual " << t->ns[h.j] << "]"; }
      }
      u.acc = (float)good / by[h.a].size();
      std::ostringstream vs; vs << (u.acc >= .99f ? "CONFIRMED" : "REFUTED") << " (" << (int)(100 * u.acc) << "% of " << by[h.a].size() << " samples within tolerance): " << h.text << (u.acc < 1.f ? "  misses:" + miss.str() : std::string());
      verdicts.push_back(vs.str());
      if (u.acc >= .99f) { if (!r[h.a][h.j].ok) nrules++; u.ok = true; r[h.a][h.j] = u; hyp_ok++; } else hyp_bad++;
    }
    ready = nrules > 0;
  }
  void write_report(const std::string& path, const std::vector<Transition>& rep) const {
    std::ofstream o(path);
    o << "# nexus synthesis report: dimensions of the observation that no verified rule explains yet.\n"
         "# Propose formulas in a rules file, one per line:   f ACTION DIM = c + c1*s2 + c2*sin(w*s0) + c3*cos(w*s0) + c4*s1*s2 + c5*s3^2 + c6*s1\n"
         "# (s<k> is observation dimension k BEFORE the step; the formula gives dimension DIM AFTER the step; run with --rules FILE and the agent confirms or refutes each line).\n";
    for (const auto& v : verdicts) o << "# last verdict: " << v << "\n";
    std::vector<const Transition*> by[ACT];
    for (const auto& t : rep) by[t.a].push_back(&t);
    for (int a = 0; a < ACT; a++) {
      if (by[a].size() < 200) continue;
      for (int j = 0; j < OBS; j++) {
        if (r[a][j].ok) continue;
        float lo = 1e9f, hi = -1e9f;
        for (const auto* t : by[a]) { lo = std::min(lo, t->ns[j]); hi = std::max(hi, t->ns[j]); }
        if (hi - lo < 1e-4f) continue;
        o << "\naction " << a << " dim " << j << ": unexplained (best library fit on held-out data " << (int)(100 * r[a][j].acc) << "%). Samples  s0..s9 -> s" << j << "':\n";
        const size_t step = std::max<size_t>(1, by[a].size() / 14);
        for (size_t i = 0; i < by[a].size() && i / step < 14; i += step) {
          o << "  ";
          for (int k = 0; k < OBS; k++) o << std::fixed << std::setprecision(4) << by[a][i]->s[k] << " ";
          o << "-> " << by[a][i]->ns[j] << "\n";
        }
      }
    }
  }
  bool off = false;  // true while the agent measures its neural model alone
  int pruned = 0;
  bool has(int a, int j) const { return !off && r[a][j].ok; }
  float next(int a, int j, float s) const { return apply1(r[a][j], s); }
  static float feat(const Term& t, const Vec& s) {
    switch (t.type) { case 0: return s[t.k]; case 1: return s[t.k] * s[t.k]; case 2: return s[t.k] * s[t.m];
                      case 3: return std::sin(t.w * s[t.k]); case 4: return std::cos(t.w * s[t.k]); default: return 0.f; }
  }
  float featA(int a, const Term& t, const Vec& s) const { return t.type == 5 ? next(a, t.k, s) : feat(t, s); }
  static float eval_lib(const Rule& u, const Vec& s, int j) {
    float v = (u.absolute ? 0.f : s[j]) + u.b0;
    for (const auto& t : u.terms) v += t.c * feat(t, s);
    if (u.has_cond) {
      float c = u.cb;
      for (const auto& t : u.ct) c += t.c * feat(t, s);
      if ((u.cdir < 0 && c < u.cthr) || (u.cdir > 0 && c > u.cthr)) v = u.cval;
    }
    return clampf(v, u.lo, u.hi);
  }
  float next(int a, int j, const Vec& s) const {
    const Rule& u = r[a][j];
    if (u.rot) {
      const int cd = u.rot == 1 ? j : u.rp, sd = u.rot == 1 ? u.rp : j;
      float d = u.b0;
      for (const auto& t : u.terms) d += t.c * featA(a, t, s);
      const float ph = std::atan2(s[sd], s[cd]) + d;
      return u.rot == 1 ? std::cos(ph) : std::sin(ph);
    }
    if (!u.lib) return apply1(u, s[j]);
    return eval_lib(u, s, j);
  }
  // rotation rules: a pair (j1, j2) that lies on the unit circle (cos/sin of an angle) is advanced by an angle increment that is fitted
  // as a short formula over the library (orthogonal matching pursuit, <= 3 terms); accepted if >= 99% of a held-out half is reproduced.
  bool fit_rot(int a, const std::vector<const Transition*>& smp, int j1, int j2, const std::vector<int>& dims, Rule& r1, Rule& r2) const {
    const int N = (int)smp.size();
    if (N < 200) return false;
    for (const auto* t : smp) {
      if (std::fabs(t->s[j1] * t->s[j1] + t->s[j2] * t->s[j2] - 1.f) > 2e-3f || std::fabs(t->ns[j1] * t->ns[j1] + t->ns[j2] * t->ns[j2] - 1.f) > 2e-3f) return false;
    }
    const int stride = std::max(1, N / 4000);
    std::vector<const Transition*> tr, te;
    for (int i = 0, c = 0; i < N; i += stride, c++) {
      if (c % 2) { te.push_back(smp[i]); continue; }
      bool sat = false;  // a saturated (|value| = 1) other dimension means a clamp was active: keep such samples out of the fit
      for (int k : dims) if (k != j1 && k != j2 && std::fabs(smp[i]->ns[k]) > .999f) sat = true;
      if (!sat) tr.push_back(smp[i]);
    }
    if (tr.size() < 100) return false;
    auto dlt = [&](const Transition* t) {
      double d = std::atan2(t->ns[j2], t->ns[j1]) - std::atan2(t->s[j2], t->s[j1]);
      while (d > M_PI) d -= 2 * M_PI;
      while (d < -M_PI) d += 2 * M_PI;
      return d;
    };
    std::vector<Term> F;
    for (int k : dims) { if (k == j1 || k == j2) { Term t; t.type = 0; t.k = k; F.push_back(t); continue; } Term t; t.type = 0; t.k = k; F.push_back(t); t.type = 1; F.push_back(t); }
    for (size_t x = 0; x < dims.size(); x++) for (size_t y2 = x + 1; y2 < dims.size(); y2++) { Term t; t.type = 2; t.k = dims[x]; t.m = dims[y2]; F.push_back(t); }
    for (int k : dims) if (k != j1 && k != j2 && r[a][k].ok && !r[a][k].rot && (r[a][k].lib || r[a][k].q > 0.f)) { Term t; t.type = 5; t.k = k; F.push_back(t); }
    const int nt = (int)tr.size();
    std::vector<double> y(nt);
    for (int i = 0; i < nt; i++) y[i] = dlt(tr[i]);
    std::vector<std::vector<double>> cols(F.size(), std::vector<double>(nt));
    for (size_t i = 0; i < F.size(); i++) for (int t = 0; t < nt; t++) cols[i][t] = featA(a, F[i], tr[t]->s);
    std::vector<int> order; std::vector<std::vector<double>> X; std::vector<double> res = y, beta;
    double ym = 0; for (double v : y) ym += v; ym /= nt;
    for (double& v : res) v -= ym;
    for (int it = 0; it < 3; it++) {
      double bestc = 0; int bi = -1;
      for (size_t i = 0; i < F.size(); i++) {
        if (std::find(order.begin(), order.end(), (int)i) != order.end()) continue;
        double m = 0; for (double v : cols[i]) m += v; m /= nt;
        double sxy = 0, sxx = 0;
        for (int t = 0; t < nt; t++) { const double d = cols[i][t] - m; sxy += d * res[t]; sxx += d * d; }
        if (sxx < 1e-10) continue;
        const double c = std::fabs(sxy) / std::sqrt(sxx);
        if (c > bestc) { bestc = c; bi = (int)i; }
      }
      if (bi < 0) break;
      order.push_back(bi); X.push_back(cols[bi]);
      if (!solve_ls(X, y, beta)) return false;
      double rms = 0;
      for (int t = 0; t < nt; t++) { double pr = beta.back(); for (size_t k = 0; k < X.size(); k++) pr += beta[k] * X[k][t]; res[t] = y[t] - pr; rms += res[t] * res[t]; }
      if (std::sqrt(rms / nt) < 1e-4) break;
    }
    if (order.empty()) return false;
    Rule u; u.lib = true; u.b0 = (float)beta.back();
    for (size_t q = 0; q < order.size(); q++) { Term t = F[order[q]]; t.c = (float)beta[q]; u.terms.push_back(t); }
    int good = 0;
    for (const auto* t : te) {
      float d = u.b0; for (const auto& tt : u.terms) d += tt.c * featA(a, tt, t->s);
      const float ph = std::atan2(t->s[j2], t->s[j1]) + d;
      if (std::fabs(std::cos(ph) - t->ns[j1]) < TOL && std::fabs(std::sin(ph) - t->ns[j2]) < TOL) good++;
    }
    const float acc = (float)good / std::max<size_t>(1, te.size());
    if (std::getenv("NEXUS_SYNTH_DEBUG")) { std::cerr << "  [fit_rot dims " << j1 << "," << j2 << " N=" << N << " train " << tr.size() << " acc " << acc << "] "; for (const auto& t : u.terms) std::cerr << term_str(t) << " "; std::cerr << "+ " << u.b0 << "\n"; }
    r1 = u; r2 = u; r1.acc = r2.acc = acc;
    if (acc < .99f) return false;
    r1.ok = r2.ok = true; r1.rot = 1; r1.rp = j2; r2.rot = 2; r2.rp = j1;
    return true;
  }
  // ---- library fit: sparse regression of the increment ns_j - s_j on {s_k, s_k^2, s_k*s_m, sin(w s_k), cos(w s_k)} (orthogonal matching pursuit) ----
  static bool solve_ls(const std::vector<std::vector<double>>& X, const std::vector<double>& y, std::vector<double>& beta) {
    const int p = (int)X.size() + 1, n = (int)y.size();  // intercept last
    std::vector<std::vector<double>> A(p, std::vector<double>(p + 1, 0.0));
    for (int i = 0; i < p; i++) {
      for (int k = 0; k < p; k++) {
        double sum = 0;
        for (int t = 0; t < n; t++) sum += (i < p - 1 ? X[i][t] : 1.0) * (k < p - 1 ? X[k][t] : 1.0);
        A[i][k] = sum + (i == k ? 1e-9 : 0.0);
      }
      double sum = 0;
      for (int t = 0; t < n; t++) sum += (i < p - 1 ? X[i][t] : 1.0) * y[t];
      A[i][p] = sum;
    }
    for (int c = 0; c < p; c++) {
      int piv = c;
      for (int r2 = c + 1; r2 < p; r2++) if (std::fabs(A[r2][c]) > std::fabs(A[piv][c])) piv = r2;
      if (std::fabs(A[piv][c]) < 1e-12) return false;
      std::swap(A[c], A[piv]);
      for (int r2 = 0; r2 < p; r2++) if (r2 != c) { const double f = A[r2][c] / A[c][c]; for (int k = c; k <= p; k++) A[r2][k] -= f * A[c][k]; }
    }
    beta.assign(p, 0.0);
    for (int i = 0; i < p; i++) beta[i] = A[i][p] / A[i][i];
    return true;
  }
  static std::string term_str(const Term& t) {
    std::ostringstream o; o << std::setprecision(4);
    o << t.c << "*";
    switch (t.type) { case 5: o << "next(s" << t.k << ")"; break; case 0: o << "s" << t.k; break; case 1: o << "s" << t.k << "^2"; break; case 2: o << "s" << t.k << "*s" << t.m; break;
                      case 3: o << "sin(" << t.w << "*s" << t.k << ")"; break; default: o << "cos(" << t.w << "*s" << t.k << ")"; }
    return o.str();
  }
  std::string describe(int a, int j) const {
    const Rule& u = r[a][j];
    std::ostringstream o; o << std::setprecision(4);
    if (u.rot) { o << "angle(s" << (u.rot == 1 ? j : u.rp) << ",s" << (u.rot == 1 ? u.rp : j) << ") advances by " << u.b0; for (const auto& t : u.terms) o << " + " << term_str(t); o << "  [" << (u.rot == 1 ? "cos" : "sin") << " component, held-out " << (int)(100 * u.acc) << "%]"; return o.str(); }
    if (!u.lib) { o << "s" << j << "' = " << u.alpha << "*s" << j << " + " << u.beta << " (guarded)"; return o.str(); }
    o << "s" << j << "' = " << (u.absolute ? "" : "s" + std::to_string(j) + " + ") << u.b0;
    for (const auto& t : u.terms) o << " + " << term_str(t);
    o << "  [clamped " << u.lo << ".." << u.hi << ", held-out " << (int)(100 * u.acc) << "%]";
    return o.str();
  }
  // fits one (action, dim); returns false if no short formula explains >= 99% of the held-out half.
  // Terms are added one by one (orthogonal matching pursuit); a sin or cos term always brings its partner and both share one refined frequency
  // (so a phase shift costs nothing); the SMALLEST prefix that reaches 99% on the held-out half is returned.
  bool fit_lib(const std::vector<const Transition*>& smp, int j, const std::vector<int>& dims, Rule& out) const {
    const int N = (int)smp.size();
    if (N < 200) return false;
    const int stride = std::max(1, N / 4000);
    std::vector<const Transition*> tr, te;
    float slo = 1e9f, shi = -1e9f;
    for (const auto* t : smp) { slo = std::min(slo, t->ns[j]); shi = std::max(shi, t->ns[j]); }
    for (int i = 0, c = 0; i < N; i += stride, c++) {
      if (c % 2) te.push_back(smp[i]);
      else if (smp[i]->ns[j] < shi - 1e-4f && smp[i]->ns[j] > slo + 1e-4f) tr.push_back(smp[i]);  // samples at the range limit may be clamped: they would bias a least-squares fit
    }
    if (tr.size() < 100) return false;
    std::vector<Term> F;
    for (int k : dims) {
      Term t; t.type = 0; t.k = k; F.push_back(t); t.type = 1; F.push_back(t);
      const float ws[] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 8.f, 10.f, 12.f, 16.f};
      for (float w : ws) { t.type = 3; t.w = w; F.push_back(t); t.type = 4; F.push_back(t); }
    }
    for (size_t x = 0; x < dims.size(); x++) for (size_t y2 = x + 1; y2 < dims.size(); y2++) { Term t; t.type = 2; t.k = dims[x]; t.m = dims[y2]; F.push_back(t); }
    const int nt = (int)tr.size();
    std::vector<double> y(nt);
    for (int i = 0; i < nt; i++) y[i] = tr[i]->ns[j] - tr[i]->s[j];
    auto column = [&](const Term& t) { std::vector<double> c(nt); for (int i = 0; i < nt; i++) c[i] = feat(t, tr[i]->s); return c; };
    std::vector<std::vector<double>> cols(F.size());
    for (size_t i = 0; i < F.size(); i++) cols[i] = column(F[i]);
    float lo = 1e9f, hi = -1e9f;
    for (const auto* t : tr) { lo = std::min(lo, t->ns[j]); hi = std::max(hi, t->ns[j]); }
    auto heldout = [&](const Rule& u) {
      int good = 0;
      for (const auto* t : te) {
        float v = t->s[j] + u.b0;
        for (const auto& tt : u.terms) v += tt.c * feat(tt, t->s);
        if (std::fabs(clampf(v, u.lo, u.hi) - t->ns[j]) < TOL) good++;
      }
      return (float)good / std::max<size_t>(1, te.size());
    };
    // greedy order of picks (on the full library)
    std::vector<int> order;
    {
      std::vector<std::vector<double>> X;
      std::vector<double> res = y, beta;
      double ym = 0; for (double v : y) ym += v; ym /= nt;
      for (double& v : res) v -= ym;
      for (int it = 0; it < 4; it++) {
        double bestc = 0; int bi = -1;
        for (size_t i = 0; i < F.size(); i++) {
          if (std::find(order.begin(), order.end(), (int)i) != order.end()) continue;
          double m = 0; for (double v : cols[i]) m += v; m /= nt;
          double sxy = 0, sxx = 0;
          for (int t = 0; t < nt; t++) { const double d = cols[i][t] - m; sxy += d * res[t]; sxx += d * d; }
          if (sxx < 1e-10) continue;
          const double c = std::fabs(sxy) / std::sqrt(sxx);
          if (c > bestc) { bestc = c; bi = (int)i; }
        }
        if (bi < 0) break;
        order.push_back(bi); X.push_back(cols[bi]);
        if (!solve_ls(X, y, beta)) break;
        double rms = 0;
        for (int t = 0; t < nt; t++) { double p = beta.back(); for (size_t k = 0; k < X.size(); k++) p += beta[k] * X[k][t]; res[t] = y[t] - p; rms += res[t] * res[t]; }
        if (std::sqrt(rms / nt) < 2e-4) break;
      }
    }
    float best_acc = 0.f;
    for (size_t m = 1; m <= order.size(); m++) {
      std::vector<Term> T;
      for (size_t q = 0; q < m; q++) {
        Term t = F[order[q]];
        T.push_back(t);
        if (t.type >= 3) {  // bring the partner (same dimension and frequency)
          Term pt = t; pt.type = t.type == 3 ? 4 : 3;
          bool have = false;
          for (size_t z = 0; z < m; z++) if (F[order[z]].type == pt.type && F[order[z]].k == pt.k && F[order[z]].w == pt.w) have = true;
          if (!have) T.push_back(pt);
        }
      }
      // remove duplicates of the same partner
      std::vector<Term> U;
      for (const auto& t : T) { bool dup = false; for (const auto& u2 : U) if (u2.type == t.type && u2.k == t.k && u2.m == t.m && u2.w == t.w) dup = true; if (!dup) U.push_back(t); }
      T = U;
      // joint frequency refinement per (dimension, frequency) group
      std::vector<std::vector<double>> X;
      for (const auto& t : T) X.push_back(column(t));
      std::vector<double> beta;
      for (size_t q = 0; q < T.size(); q++) {
        if (T[q].type != 3) continue;
        const float w0 = T[q].w; const int kk = T[q].k;
        double bw = w0, be = 1e30;
        for (int g = -30; g <= 30; g++) {
          const float w = (float)(w0 * (1.0 + g * 0.01));
          std::vector<std::vector<double>> X2 = X;
          for (size_t z = 0; z < T.size(); z++) if (T[z].type >= 3 && T[z].k == kk && T[z].w == w0) { Term u = T[z]; u.w = w; X2[z] = column(u); }
          std::vector<double> b2;
          if (!solve_ls(X2, y, b2)) continue;
          double e = 0;
          for (int i = 0; i < nt; i++) { double pr = b2.back(); for (size_t k = 0; k < X2.size(); k++) pr += b2[k] * X2[k][i]; e += (y[i] - pr) * (y[i] - pr); }
          if (e < be) { be = e; bw = w; }
        }
        for (size_t z = 0; z < T.size(); z++) if (T[z].type >= 3 && T[z].k == kk && T[z].w == w0) { T[z].w = (float)bw; X[z] = column(T[z]); }
      }
      if (!solve_ls(X, y, beta)) continue;
      Rule u; u.lib = true; u.b0 = (float)beta.back(); u.lo = -1.f; u.hi = 1.f;  // not the observed range: that would forbid predicting unseen (goal) regions
      for (size_t q = 0; q < T.size(); q++) { Term t = T[q]; t.c = (float)beta[q]; u.terms.push_back(t); }
      u.acc = heldout(u);
      if (std::getenv("NEXUS_SYNTH_DEBUG")) { std::cerr << "  [fit_lib dim " << j << " N=" << N << " prefix " << m << " acc " << u.acc << "] "; for (const auto& t : u.terms) std::cerr << term_str(t) << " "; std::cerr << "+ " << u.b0 << "\n"; }
      best_acc = std::max(best_acc, u.acc);
      if (u.acc >= .99f) { u.ok = true; out = u; return true; }
    }
    out.acc = best_acc;
    return false;
  }
  std::string summary() const {
    std::ostringstream o;
    int nontriv = 0;
    for (int a = 0; a < ACT; a++) for (int j = 0; j < OBS; j++) if (r[a][j].ok && r[a][j].q > 0.f) nontriv++;
    o << "synth: " << nrules << " verified rules (" << pruned << " pruned for not beating the network) (" << nontriv << " that change a value, " << nlib << " from the formula library), rules file: " << hyp_ok << " confirmed, " << hyp_bad << " refuted/unsupported";
    return o.str();
  }
};

struct TokenModel {
  int K = 64, D = 16, H = 96, RH = 64;
  bool ready = false;
  std::vector<int> ad;                         // active observation dimensions (they vary in the data)
  std::vector<int> pd;                         // positions in ad that the network predicts (the others are derived)
  std::array<bool, OBS> der{};                 // derived dimension: an exact linear function of other dimensions (found from data)
  std::array<std::array<float, OBS + 1>, OBS> dco{};  // its coefficients (last entry: constant)
  std::array<std::vector<float>, OBS> cb, cn;  // sorted centroids and their counts
  std::array<std::vector<float>, OBS> emb, gemb, memb, vemb;
  uint64_t et = 0;
  Lin l1, l2, ld, r1, r2;
  mutable std::vector<float> sx, sh1, sh2, slg, sxr, shr;  // scratch (single-threaded)

  int na() const { return (int)ad.size(); }
  int tok(int j, float x) const {  // nearest centroid
    const auto& c = cb[j];
    const int k = (int)(std::lower_bound(c.begin(), c.end(), x) - c.begin());
    if (k <= 0) return 0;
    if (k >= K) return K - 1;
    return (x - c[k - 1] <= c[k] - x) ? k - 1 : k;
  }
  // Exact linear dependencies between dimensions (e.g. 'distance to the target' = target - current): least squares on the samples that are not
  // clamped; a dimension is derived when the largest residual is below half a cell. Derived dimensions are recomputed, not predicted.
  void find_derived(const std::vector<Transition>& rp) {
    der.fill(false);
    for (auto& r : dco) r.fill(0.f);
    if (rp.size() < 200) return;
    for (int ii = (int)ad.size() - 1; ii >= 0; ii--) {
      const int j = ad[ii];
      std::vector<int> reg;
      for (int k : ad) if (k != j && !der[k]) reg.push_back(k);
      const int m = (int)reg.size();
      if (m == 0) continue;
      std::vector<std::vector<double>> A((size_t)m + 1, std::vector<double>((size_t)m + 2, 0.0));
      std::vector<const Vec*> use;
      for (const auto& z : rp) for (const Vec* v : {&z.s, &z.ns}) {
        bool ok = std::abs((*v)[j]) < .98f;
        for (int k : reg) ok = ok && std::abs((*v)[k]) < .98f;
        if (ok) use.push_back(v);
      }
      if (use.size() < 200) continue;
      for (const Vec* v : use) {
        std::vector<double> f((size_t)m + 1, 1.0);
        for (int a = 0; a < m; a++) f[a] = (*v)[reg[a]];
        for (int a = 0; a <= m; a++) { for (int b = 0; b <= m; b++) A[a][b] += f[a] * f[b]; A[a][m + 1] += f[a] * (*v)[j]; }
      }
      for (int a = 0; a <= m; a++) A[a][a] += 1e-6 * (double)use.size();
      bool sing = false;
      for (int c = 0; c <= m && !sing; c++) {  // Gauss-Jordan with partial pivoting
        int pv = c; for (int r = c + 1; r <= m; r++) if (std::abs(A[r][c]) > std::abs(A[pv][c])) pv = r;
        if (std::abs(A[pv][c]) < 1e-12) { sing = true; break; }
        std::swap(A[c], A[pv]);
        for (int r = 0; r <= m; r++) if (r != c) { const double fct = A[r][c] / A[c][c]; for (int k = c; k <= m + 1; k++) A[r][k] -= fct * A[c][k]; }
      }
      if (sing) continue;
      std::array<float, OBS + 1> co{}; for (int a = 0; a < m; a++) co[reg[a]] = (float)(A[a][m + 1] / A[a][a]); co[OBS] = (float)(A[m][m + 1] / A[m][m]);
      double worst = 0;
      for (const Vec* v : use) { double pr = co[OBS]; for (int k : reg) pr += (double)co[k] * (*v)[k]; worst = std::max(worst, std::abs(pr - (*v)[j])); }
      if (worst < .5 * 2.0 / (double)K) { der[j] = true; dco[j] = co; }
    }
  }
  void init(const std::vector<Transition>& rp, int K_, uint32_t seed) {
    K = K_; ad.clear();
    for (int j = 0; j < OBS; j++) {
      float lo = 1e9f, hi = -1e9f;
      for (const auto& z : rp) { lo = std::min(lo, std::min(z.s[j], z.ns[j])); hi = std::max(hi, std::max(z.s[j], z.ns[j])); }
      if (hi - lo > 1e-4f) ad.push_back(j);
      cb[j].assign((size_t)K, 0.f); cn[j].assign((size_t)K, 1.f);
      for (int k = 0; k < K; k++) cb[j][k] = -1.f + 2.f * (float)k / (float)K;  // uniform grid: exact for dimensions that live on a grid of 2/K
      emb[j].clear(); gemb[j].clear(); memb[j].clear(); vemb[j].clear();
    }
    std::mt19937 r(seed);
    std::normal_distribution<float> nd(0.f, .3f);
    for (int j : ad) {
      emb[j].assign((size_t)K * D, 0.f); for (auto& x : emb[j]) x = nd(r);
      gemb[j].assign(emb[j].size(), 0.f); memb[j] = vemb[j] = gemb[j];
    }
    find_derived(rp);
    pd.clear(); for (int i = 0; i < na(); i++) if (!der[ad[i]]) pd.push_back(i);
    const int n = na(), np = (int)pd.size();
    l1.init(n * D + ACT, H, seed + 1); l2.init(H, H, seed + 2); ld.init(H, np * K, seed + 3, .1f);
    r1.init(2 * n * D + ACT, RH, seed + 4); r2.init(RH, 1, seed + 5, .1f);
    for (const auto& z : rp) { observe(z.s); observe(z.ns); }
    et = 0; ready = true;
  }
  void observe(const Vec& s) {  // online k-means step; a centroid moves inside its own cell, so the order is preserved
    for (int j : ad) {
      const int k = tok(j, s[j]);
      cn[j][k] += 1.f;
      cb[j][k] += (s[j] - cb[j][k]) / std::min(cn[j][k], 100.f);
    }
  }
  void toks(const Vec& s, int* t) const { for (int i = 0; i < na(); i++) t[i] = tok(ad[i], s[ad[i]]); }
  void put_emb(const int* t, float* x) const { for (int i = 0; i < na(); i++) std::copy_n(&emb[ad[i]][(size_t)t[i] * D], D, x + i * D); }
  void scratch() const {
    const int n = na();
    sx.assign((size_t)n * D + ACT, 0.f); sh1.assign((size_t)H, 0.f); sh2.assign((size_t)H, 0.f); slg.assign((size_t)pd.size() * K, 0.f);
    sxr.assign((size_t)2 * n * D + ACT, 0.f); shr.assign((size_t)RH, 0.f);
  }
  void fwd_dyn(const int* t, int a) const {  // fills sx, sh1, sh2, slg
    const int n = na();
    std::fill(sx.begin(), sx.end(), 0.f);
    put_emb(t, sx.data()); sx[(size_t)n * D + a] = 1.f;
    l1.forward(sx.data(), sh1.data()); for (auto& v : sh1) v = std::tanh(v);
    l2.forward(sh1.data(), sh2.data()); for (auto& v : sh2) v = std::tanh(v);
    ld.forward(sh2.data(), slg.data());
  }
  float fwd_rew(const int* t, const int* tn, int a) const {  // reward from the tokens of the state and of the next state
    const int n = na();
    std::fill(sxr.begin(), sxr.end(), 0.f);
    put_emb(tn, sxr.data()); put_emb(t, sxr.data() + n * D); sxr[(size_t)2 * n * D + a] = 1.f;
    r1.forward(sxr.data(), shr.data()); for (auto& v : shr) v = std::tanh(v);
    float o = 0.f; r2.forward(shr.data(), &o);
    return o;
  }
  void predict(const Vec& s, int a, Vec& ns, float& r, const Synth* sy = nullptr) const {
    if (sx.empty()) scratch();
    int t[OBS], tn[OBS];
    toks(s, t);
    fwd_dyn(t, a);
    ns = s;
    for (size_t q = 0; q < pd.size(); q++) {
      const float* lg = &slg[q * K];
      const int best = (int)(std::max_element(lg, lg + K) - lg), i = pd[q];
      tn[i] = best; ns[ad[i]] = cb[ad[i]][best];
    }
    if (sy && sy->ready) for (int i = 0; i < na(); i++) if (!der[ad[i]] && sy->has(a, ad[i])) { ns[ad[i]] = sy->next(a, ad[i], s); tn[i] = tok(ad[i], ns[ad[i]]); }
    for (int i = 0; i < na(); i++) if (der[ad[i]]) {  // derived dimensions follow from the predicted ones
      float v = dco[ad[i]][OBS];
      for (int k = 0; k < OBS; k++) if (dco[ad[i]][k] != 0.f) v += dco[ad[i]][k] * ns[k];
      ns[ad[i]] = clampf(v, -1.f, 1.f); tn[i] = tok(ad[i], ns[ad[i]]);
    }
    r = clampf(fwd_rew(t, tn, a), -1.f, 1.f) * REWARD_SCALE;
  }
  // forward + backward for one transition; returns the loss (cross-entropy summed over dimensions + weighted reward error)
  double accumulate(const Transition& z, bool backprop) {
    if (sx.empty()) scratch();
    const int n = na();
    int t[OBS], tn[OBS];
    toks(z.s, t); toks(z.ns, tn);
    fwd_dyn(t, z.a);
    double loss = 0;
    std::vector<float> dl(pd.size() * (size_t)K, 0.f);
    for (size_t q = 0; q < pd.size(); q++) {
      const float* lg = &slg[q * K];
      const int tgt = tn[pd[q]];
      const float mx = *std::max_element(lg, lg + K);
      double zsum = 0; for (int k = 0; k < K; k++) zsum += std::exp((double)(lg[k] - mx));
      loss += -((double)(lg[tgt] - mx) - std::log(zsum));
      for (int k = 0; k < K; k++) dl[q * K + k] = (float)(std::exp((double)(lg[k] - mx)) / zsum);
      dl[q * K + tgt] -= 1.f;
    }
    const float tr = clampf(z.r / REWARD_SCALE, -1.f, 1.f);
    const float ro = fwd_rew(t, tn, z.a), err = ro - tr;
    loss += .5 * cfg.rw * (double)err * err;
    if (!backprop) return loss;
    std::vector<float> dh2((size_t)H, 0.f), dh1((size_t)H, 0.f), dx(sx.size(), 0.f);
    ld.backward(sh2.data(), dl.data(), dh2.data());
    for (int k = 0; k < H; k++) dh2[k] *= 1.f - sh2[k] * sh2[k];
    l2.backward(sh1.data(), dh2.data(), dh1.data());
    for (int k = 0; k < H; k++) dh1[k] *= 1.f - sh1[k] * sh1[k];
    l1.backward(sx.data(), dh1.data(), dx.data());
    for (int i = 0; i < n; i++) for (int d = 0; d < D; d++) gemb[ad[i]][(size_t)t[i] * D + d] += dx[(size_t)i * D + d];
    float dro = cfg.rw * err;
    std::vector<float> dhr((size_t)RH, 0.f), dxr(sxr.size(), 0.f);
    r2.backward(shr.data(), &dro, dhr.data());
    for (int k = 0; k < RH; k++) dhr[k] *= 1.f - shr[k] * shr[k];
    r1.backward(sxr.data(), dhr.data(), dxr.data());
    for (int i = 0; i < n; i++) for (int d = 0; d < D; d++) {
      gemb[ad[i]][(size_t)tn[i] * D + d] += dxr[(size_t)i * D + d];
      gemb[ad[i]][(size_t)t[i] * D + d] += dxr[(size_t)(n + i) * D + d];
    }
    return loss;
  }
  void step_all(float lr, float scale) {
    l1.step(lr, scale); l2.step(lr, scale); ld.step(lr, scale); r1.step(lr, scale); r2.step(lr, scale);
    et++;
    const float c1 = 1.f - std::pow(.9f, (float)et), c2 = 1.f - std::pow(.999f, (float)et);
    for (int j : ad) for (size_t i = 0; i < emb[j].size(); i++) {
      const float g = gemb[j][i] * scale;
      memb[j][i] = .9f * memb[j][i] + .1f * g; vemb[j][i] = .999f * vemb[j][i] + .001f * g * g;
      emb[j][i] -= lr * (memb[j][i] / c1) / (std::sqrt(vemb[j][i] / c2) + 1e-8f);
      gemb[j][i] = 0.f;
    }
  }
  void save(std::ostream& o) const {
    wr(o, (uint8_t)ready); wr(o, K); wr(o, D); wr(o, H); wr(o, RH); wr(o, et);
    int n = na(); wr(o, n); for (int j : ad) wr(o, j);
    if (!ready) return;
    for (int j = 0; j < OBS; j++) { wr(o, (uint8_t)der[j]); for (float x : dco[j]) wr(o, x); }
    for (int j : ad) { Lin::putv(o, cb[j]); Lin::putv(o, cn[j]); Lin::putv(o, emb[j]); Lin::putv(o, memb[j]); Lin::putv(o, vemb[j]); }
    l1.save(o); l2.save(o); ld.save(o); r1.save(o); r2.save(o);
  }
  bool load(std::istream& i) {
    TokenModel T;
    uint8_t rdy = 0; int n = 0;
    rd(i, rdy); rd(i, T.K); rd(i, T.D); rd(i, T.H); rd(i, T.RH); rd(i, T.et); rd(i, n);
    if (!i || T.K < 2 || T.K > 4096 || T.D < 1 || T.D > 64 || T.H < 1 || T.H > 1024 || T.RH < 1 || T.RH > 1024 || n < 0 || n > OBS) return false;
    for (int k = 0; k < n; k++) { int j = 0; rd(i, j); if (!i || j < 0 || j >= OBS) return false; T.ad.push_back(j); }
    T.ready = rdy != 0;
    if (T.ready) {
      for (int j = 0; j < OBS; j++) { uint8_t dd = 0; rd(i, dd); T.der[j] = dd != 0; for (float& x : T.dco[j]) { rd(i, x); if (!std::isfinite(x)) return false; } }
      if (!i) return false;
      for (int k = 0; k < (int)T.ad.size(); k++) if (!T.der[T.ad[k]]) T.pd.push_back(k);
      for (int j : T.ad) {
        if (!Lin::getv(i, T.cb[j], (uint64_t)T.K) || !Lin::getv(i, T.cn[j], (uint64_t)T.K)) return false;
        const uint64_t ne = (uint64_t)T.K * T.D;
        if (!Lin::getv(i, T.emb[j], ne) || !Lin::getv(i, T.memb[j], ne) || !Lin::getv(i, T.vemb[j], ne)) return false;
        T.gemb[j].assign((size_t)ne, 0.f);
      }
      const int m = (int)T.ad.size(), mp = (int)T.pd.size();
      T.l1.init(m * T.D + ACT, T.H, 0); T.l2.init(T.H, T.H, 0); T.ld.init(T.H, mp * T.K, 0); T.r1.init(2 * m * T.D + ACT, T.RH, 0); T.r2.init(T.RH, 1, 0);
      if (!T.l1.load(i) || !T.l2.load(i) || !T.ld.load(i) || !T.r1.load(i) || !T.r2.load(i)) return false;
    }
    *this = T;
    return true;
  }
};

// ---------------------------------------------------------------------------------------------
// v9 / item 2: a manager above the CEM planner. "Concepts" are clusters of visited states (k-means in a z-scored space); the manager picks
// the next concept to go to (an option) from a learned option value Q_hi(s, concept); the low level is the existing planner with a potential-based
// reward for getting closer to the chosen concept. An option ends when the concept is reached or after hier_T primitive steps.
// ---------------------------------------------------------------------------------------------
struct Concepts {
  int NC = 0;
  bool ready = false;
  std::vector<int> dims;
  std::array<float, OBS> mu{}, sd{};
  std::vector<Vec> c;
  std::vector<float> cnt, rad;
  float dist(const Vec& a, const Vec& b) const {  // RMS z-distance over the used dimensions
    float e = 0;
    for (int j : dims) e += sq((a[j] - b[j]) / sd[j]);
    return std::sqrt(e / (float)std::max<size_t>(1, dims.size()));
  }
  int nearest(const Vec& s, float* d = nullptr) const {
    int best = 0; float bd = 1e30f;
    for (int k = 0; k < NC; k++) { const float e = dist(s, c[k]); if (e < bd) { bd = e; best = k; } }
    if (d) *d = bd;
    return best;
  }
  void init(const std::vector<Transition>& rp, int nc, uint32_t seed) {
    NC = nc; dims.clear(); mu.fill(0.f); sd.fill(1.f);
    std::mt19937 r(seed);
    std::vector<const Vec*> pts;
    const size_t cap = 4000;
    for (size_t i = 0; i < rp.size(); i++) if (pts.size() < cap) pts.push_back(&rp[i].ns); else { const size_t k = r() % (i + 1); if (k < cap) pts[k] = &rp[i].ns; }
    for (int j = 0; j < OBS; j++) {
      double m = 0, m2 = 0; float lo = 1e9f, hi = -1e9f;
      for (auto* p : pts) { m += (*p)[j]; m2 += sq((*p)[j]); lo = std::min(lo, (*p)[j]); hi = std::max(hi, (*p)[j]); }
      m /= (double)pts.size(); const double var = std::max(0.0, m2 / (double)pts.size() - m * m);
      if (hi - lo > 1e-4f) { dims.push_back(j); mu[j] = (float)m; sd[j] = std::max(.05f, (float)std::sqrt(var)); }
    }
    c.assign((size_t)NC, Vec{}); cnt.assign((size_t)NC, 1.f); rad.assign((size_t)NC, .3f);
    c[0] = *pts[r() % pts.size()];
    std::vector<float> md(pts.size(), 1e30f);
    for (int k = 1; k < NC; k++) {  // k-means++ seeding
      double tot = 0;
      for (size_t i = 0; i < pts.size(); i++) { md[i] = std::min(md[i], dist(*pts[i], c[k - 1])); tot += sq(md[i]); }
      double u = std::uniform_real_distribution<double>(0.0, std::max(tot, 1e-12))(r), acc = 0; size_t pick = 0;
      for (size_t i = 0; i < pts.size(); i++) { acc += sq(md[i]); if (acc >= u) { pick = i; break; } }
      c[k] = *pts[pick];
    }
    for (int it = 0; it < 5; it++) {  // Lloyd
      std::vector<Vec> sum((size_t)NC, Vec{}); std::vector<float> n((size_t)NC, 0.f), dd((size_t)NC, 0.f);
      for (auto* p : pts) { float d; const int k = nearest(*p, &d); n[k] += 1.f; dd[k] += d; for (int j = 0; j < OBS; j++) sum[k][j] += (*p)[j]; }
      for (int k = 0; k < NC; k++) if (n[k] > 0.f) { for (int j = 0; j < OBS; j++) c[k][j] = sum[k][j] / n[k]; cnt[k] = n[k]; rad[k] = std::max(.15f, dd[k] / n[k]); }
    }
    ready = true;
  }
  void observe(const Vec& s) {
    float d; const int k = nearest(s, &d);
    cnt[k] += 1.f;
    const float lr = 1.f / std::min(cnt[k], 300.f);
    for (int j = 0; j < OBS; j++) c[k][j] += lr * (s[j] - c[k][j]);
    rad[k] += lr * (std::max(.15f, d) - rad[k]);
  }
  void save(std::ostream& o) const {
    wr(o, (uint8_t)ready); wr(o, NC);
    if (!ready) return;
    uint32_t nd = (uint32_t)dims.size(); wr(o, nd); for (int j : dims) wr(o, j);
    put(o, mu); put(o, sd);
    for (int k = 0; k < NC; k++) { put(o, c[k]); wr(o, cnt[k]); wr(o, rad[k]); }
  }
  bool load(std::istream& i) {
    Concepts T; uint8_t rdy = 0;
    rd(i, rdy); rd(i, T.NC);
    if (!i || T.NC < 0 || T.NC > 4096) return false;
    T.ready = rdy != 0;
    if (T.ready) {
      uint32_t nd = 0; rd(i, nd);
      if (!i || nd > (uint32_t)OBS) return false;
      for (uint32_t k = 0; k < nd; k++) { int j = 0; rd(i, j); if (!i || j < 0 || j >= OBS) return false; T.dims.push_back(j); }
      get(i, T.mu); get(i, T.sd);
      T.c.assign((size_t)T.NC, Vec{}); T.cnt.assign((size_t)T.NC, 1.f); T.rad.assign((size_t)T.NC, .3f);
      for (int k = 0; k < T.NC; k++) { get(i, T.c[k]); rd(i, T.cnt[k]); rd(i, T.rad[k]); }
      if (!i) return false;
      for (int j : T.dims) if (!(T.sd[j] > 0.f) || !std::isfinite(T.sd[j])) return false;
    }
    *this = T;
    return true;
  }
};

struct QHi {  // option value network: state -> value of every concept
  Lin a, b, c;
  std::vector<float> h1, h2;
  void init(int nc, uint32_t seed) { a.init(OBS, 64, seed); b.init(64, 64, seed + 1); c.init(64, nc, seed + 2, .1f); h1.assign(64, 0.f); h2.assign(64, 0.f); }
  void forward(const Vec& s, float* q) {
    a.forward(s.data(), h1.data()); for (auto& v : h1) v = std::tanh(v);
    b.forward(h1.data(), h2.data()); for (auto& v : h2) v = std::tanh(v);
    c.forward(h2.data(), q);
  }
  void backward(const Vec& s, const float* dq) {  // after forward(s)
    std::vector<float> d2(64, 0.f), d1(64, 0.f);
    c.backward(h2.data(), dq, d2.data());
    for (int k = 0; k < 64; k++) d2[k] *= 1.f - h2[k] * h2[k];
    b.backward(h1.data(), d2.data(), d1.data());
    for (int k = 0; k < 64; k++) d1[k] *= 1.f - h1[k] * h1[k];
    a.backward(s.data(), d1.data(), nullptr);
  }
  void step(float lr, float scale) { a.step(lr, scale); b.step(lr, scale); c.step(lr, scale); }
  void copy_from(const QHi& o) { a = o.a; b = o.b; c = o.c; }
  void save(std::ostream& o) const { a.save(o); b.save(o); c.save(o); }
  bool load(std::istream& i) { return a.load(i) && b.load(i) && c.load(i); }
};
struct OptTr { Vec s0{}, s1{}; int g = 0; float R = 0, disc = 1; bool done = false; };

// ---------------------------------------------------------------------------------------------
// v9 / item 4: persistent memory. A transition graph over concepts (nodes) with decaying edge weights, mean rewards and terminal rates per
// (node, action); value iteration on it gives a value that REASONS over remembered structure (if a road stops working its edge weight fades
// and the value switches to the detour). The replay ring no longer erases everything old: one real exemplar per graph edge is pinned.
// ---------------------------------------------------------------------------------------------
struct Graph {
  Concepts cc;
  int NC = 0;
  bool ready = false;
  struct Edge { int to; float w; };
  std::vector<std::vector<Edge>> adj;     // index node * ACT + action
  std::vector<float> rm, rc, tm_;         // mean reward, count, terminal rate
  std::vector<float> V;
  void init(const std::vector<Transition>& rp, int nc, uint32_t seed) {
    cc.init(rp, nc, seed); NC = nc;
    adj.assign((size_t)NC * ACT, {}); rm.assign((size_t)NC * ACT, 0.f); rc = rm; tm_ = rm; V.assign((size_t)NC, 0.f);
    ready = true;
  }
  int node(const Vec& s) const { return cc.nearest(s); }
  void observe_nodes(int u, int v, int a, float r, bool done) {
    const size_t k = (size_t)u * ACT + a;
    auto& es = adj[k];
    for (auto& e : es) e.w *= .995f;  // a road that is no longer taken fades
    bool found = false;
    for (auto& e : es) if (e.to == v) { e.w += 1.f; found = true; }
    if (!found) es.push_back({v, 1.f});
    es.erase(std::remove_if(es.begin(), es.end(), [](const Edge& e) { return e.w < .02f; }), es.end());
    rc[k] += 1.f;
    const float lr = 1.f / std::min(rc[k], 50.f);
    rm[k] += lr * (r - rm[k]); tm_[k] += lr * ((done ? 1.f : 0.f) - tm_[k]);
  }
  void observe(const Transition& t) { observe_nodes(node(t.s), node(t.ns), t.a, t.r, t.done); }
  float qval(int u, int a, float gamma) const {
    const size_t k = (size_t)u * ACT + a;
    if (adj[k].empty()) return -1e9f;
    float tot = 0, ev = 0;
    for (const auto& e : adj[k]) { tot += e.w; ev += e.w * V[e.to]; }
    return rm[k] + gamma * (1.f - tm_[k]) * ev / tot;
  }
  void solve(float gamma, int sweeps = 60) {
    for (int it = 0; it < sweeps; it++) {
      float change = 0;
      for (int u = 0; u < NC; u++) {
        float best = -1e9f;
        for (int a = 0; a < ACT; a++) best = std::max(best, qval(u, a, gamma));
        if (best > -1e8f) { change = std::max(change, std::abs(best - V[u])); V[u] = best; }
      }
      if (change < 1e-4f) break;
    }
  }
  int best_action(int u, float gamma) const {
    int ba = 0; float bq = -1e30f;
    for (int a = 0; a < ACT; a++) { const float q = qval(u, a, gamma); if (q > bq) { bq = q; ba = a; } }
    return ba;
  }
  float value(const Vec& s) const { return V[(size_t)node(s)]; }
  void save(std::ostream& o) const {
    wr(o, (uint8_t)ready); wr(o, NC);
    if (!ready) return;
    cc.save(o);
    for (size_t k = 0; k < adj.size(); k++) {
      uint32_t n = (uint32_t)adj[k].size(); wr(o, n);
      for (const auto& e : adj[k]) { wr(o, e.to); wr(o, e.w); }
      wr(o, rm[k]); wr(o, rc[k]); wr(o, tm_[k]);
    }
    for (float v : V) wr(o, v);
  }
  bool load(std::istream& i) {
    Graph G; uint8_t rdy = 0;
    rd(i, rdy); rd(i, G.NC);
    if (!i || G.NC < 0 || G.NC > 4096) return false;
    G.ready = rdy != 0;
    if (G.ready) {
      if (!G.cc.load(i) || G.cc.NC != G.NC) return false;
      G.adj.assign((size_t)G.NC * ACT, {}); G.rm.assign(G.adj.size(), 0.f); G.rc = G.rm; G.tm_ = G.rm; G.V.assign((size_t)G.NC, 0.f);
      for (size_t k = 0; k < G.adj.size(); k++) {
        uint32_t n = 0; rd(i, n);
        if (!i || n > (uint32_t)G.NC) return false;
        for (uint32_t q = 0; q < n; q++) { Edge e{}; rd(i, e.to); rd(i, e.w); if (!i || e.to < 0 || e.to >= G.NC || !std::isfinite(e.w)) return false; G.adj[k].push_back(e); }
        rd(i, G.rm[k]); rd(i, G.rc[k]); rd(i, G.tm_[k]);
      }
      for (auto& v : G.V) rd(i, v);
      if (!i) return false;
    }
    *this = G;
    return true;
  }
};

using Seq = std::array<int, MAXH>;
using Probs = std::array<std::array<float, ACT>, MAXH>;
struct Plan { Seq a{}; float score = -1e9f; };
struct Diag { int pos = 0; float rpos = 0, rneg = 0, qmean = 0, qgoal = 0; };

struct Agent {
  std::vector<MLP> avg_models;  // weight averages used for evaluation when --avg is set (acting and learning in training use the online weights)
  QNet avg_q;
  bool avg_ready = false, avg_in = false;
  QNet en;                          // v9 item 3: predicted world-model error per action ("where my world breaks")
  float err_ema = 0.f;
  Graph gr;                         // v9 item 4: graph memory and pinned exemplars
  std::vector<uint8_t> pin;
  std::unordered_map<uint64_t, uint32_t> pinmap;
  size_t pin_count = 0, ring = 0;
  Concepts cn;                      // v9 item 2: concepts, manager, option bookkeeping
  QHi qh, qht;
  std::vector<OptTr> ob;
  int opt_g = -1, opt_left = 0, opt_k = 0;
  bool opt_train = false;
  Vec opt_s0{}, last_s{};
  float opt_R = 0, opt_disc = 1;
  uint64_t hsteps = 0;
  mutable Vec hg{};                 // current low-level target
  mutable bool hg_on = false;
  std::array<float, 512> bn_s{}, bn_f{};  // episodes (successful / failed) in which each concept was visited
  float bn_ns = 0, bn_nf = 0;
  std::vector<char> ep_seen;               // concepts visited in the current episode
  bool ep_succ = false;
  std::array<char, 512> bn_ok{};           // sub-goal candidates
  Synth sy;                 // v10: verified symbolic rules
  TokenModel tm;            // v9: discrete symbols and next-token model (used when --vq K)
  std::vector<MLP> models;  // heap storage: the ensemble is several MB at width 96+ and must not live on the stack
  RndNet rnd_t, rnd_p;  // frozen random target, trained predictor
  QNet qn, qt;          // online / target Q-network
  RwNet rn;             // optional reward net on distance features (--rnet)
  std::vector<Transition> replay;
  std::vector<uint32_t> pos_idx;  // replay indices of goal-reaching transitions
  std::mt19937 rng;
  uint64_t steps = 0, updates = 0, episodes_done = 0, goals = 0;
  float last_loss = 0, last_qloss = 0, rnd_ema = 0;
  float lr_mult() const {
    if (cfg.lr_to <= cfg.lr_from) return 1.f;
    const float t = ((float)episodes_done - (float)cfg.lr_from) / (float)(cfg.lr_to - cfg.lr_from);
    return 1.f - .9f * clampf(t, 0.f, 1.f);
  }
  bool rnd_ready = false;
  Probs warm{};
  bool has_warm = false;
  int warm_age = 0;
  std::vector<Plan> pop;

  explicit Agent(uint32_t seed = 123) : models{MLP(seed + 1), MLP(seed + 2), MLP(seed + 3)}, rng(seed) {
    replay.reserve(REPLAY_CAP);
    rnd_t.init(seed + 11, 1.f);
    rnd_p.init(seed + 12, 1.f);
    qn.init(seed + 21, .1f);
    qt = qn;
    rn.init(seed + 31, .1f);
    en.init(seed + 41, .1f);
  }
  void bn_finish_episode() {
    if (!cfg.bottleneck || !cn.ready) { std::fill(ep_seen.begin(), ep_seen.end(), 0); ep_succ = false; return; }
    bool any = false;
    for (char c : ep_seen) if (c) { any = true; break; }
    if (any) {
      for (size_t k = 0; k < ep_seen.size() && k < 512; k++) if (ep_seen[k]) (ep_succ ? bn_s : bn_f)[k] += 1.f;
      (ep_succ ? bn_ns : bn_nf) += 1.f;
    }
    std::fill(ep_seen.begin(), ep_seen.end(), 0); ep_succ = false;
    bn_mask();
  }
  void bn_finish_episode_stats_only(float, float) { bn_mask(); }
  void bn_mask() {
    bn_ok.fill(0);
    if (bn_ns < 3.f) return;  // not enough successes to say what leads to success
    std::vector<std::pair<float, int>> lift;
    for (int k = 0; k < cn.NC && k < 512; k++) {
      const float l = (bn_s[k] + 1.f) / (bn_ns + 2.f) - (bn_f[k] + 1.f) / (bn_nf + 2.f);
      if (l > .15f) lift.push_back({l, k});
    }
    std::sort(lift.begin(), lift.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    for (size_t i = 0; i < lift.size() && i < 4; i++) bn_ok[lift[i].second] = 1;
  }
  void begin_episode() {
    bn_finish_episode();
    if (cfg.hier > 0 && opt_g >= 0) end_option(last_s, false);
    has_warm = false; warm_age = 0;
  }

  // ---- model ----
  void predict(const Vec& s, const Ctx& c, int a, Vec& ns, float& r, float& unc) const {
    if (cfg.vq > 0 && tm.ready) { tm.predict(s, a, ns, r, cfg.synth ? &sy : nullptr); unc = 0.f; return; }
    const In x = make_in(s, c);
    std::array<Out, ENSEMBLE> p{};
    Out mean{};
    for (int m = 0; m < ENSEMBLE; m++) {
      p[m] = models[m].forward(x, a, c.pa);
      for (int k = 0; k < OUT; k++) mean[k] += p[m][k] / ENSEMBLE;
    }
    for (int j = 0; j < OBS; j++) ns[j] = clampf(s[j] + mean[j] * DSCALE[j], -1.f, 1.f);
    if (cfg.synth && sy.ready) for (int j = 0; j < OBS; j++) if (sy.has(a, j)) ns[j] = sy.next(a, j, s);
    r = clampf(mean[OBS], -1.f, 1.f) * REWARD_SCALE;
    unc = 0;
    for (int m = 0; m < ENSEMBLE; m++)
      for (int j = 0; j < OBS; j++) unc += sq(p[m][j] - mean[j]);
    unc /= ENSEMBLE * OBS;
  }
  // A verified rule is only worth using where it beats the neural model: measure both on stored transitions and drop value-changing rules that
  // are not at least twice as accurate (RMS) as the network, which is often already accurate to ~1e-3 (an approximate rule would make it worse).
  void synth_prune() {
    sy.pruned = 0;
    if (!sy.ready || replay.size() < 500) return;
    for (int a = 0; a < ACT; a++) {
      std::vector<const Transition*> smp;
      const size_t stride = std::max<size_t>(1, replay.size() / 4000);
      for (size_t i = 0; i < replay.size() && smp.size() < 600; i += stride) if (replay[i].a == a) smp.push_back(&replay[i]);
      if (smp.size() < 60) continue;
      std::array<double, OBS> en{}, er{};
      sy.off = true;
      for (const auto* t : smp) { Vec ns{}; float rr, u; predict(t->s, t->c, a, ns, rr, u); for (int j = 0; j < OBS; j++) en[j] += sq(ns[j] - t->ns[j]); }
      sy.off = false;
      for (const auto* t : smp) for (int j = 0; j < OBS; j++) if (sy.r[a][j].ok) er[j] += sq(sy.next(a, j, t->s) - t->ns[j]);
      for (int j = 0; j < OBS; j++) {
        Synth::Rule& u = sy.r[a][j];
        if (!u.ok || !(u.lib || u.q > 0.f)) continue;
        const double rn = std::sqrt(en[j] / smp.size()), rr = std::sqrt(er[j] / smp.size());
        if (rr > 1e-4 && rr > .5 * rn) { u.ok = false; sy.nrules--; sy.pruned++; if (u.rot) sy.r[a][u.rp].ok = false; }
      }
    }
  }
  float state_error(const Transition& t) const {
    Vec ns{};
    float r, u;
    predict(t.s, t.c, t.a, ns, r, u);
    float e = 0;
    for (int j = 0; j < OBS; j++) e += sq(ns[j] - t.ns[j]);
    return e / OBS;
  }

  // ---- replay ----
  static bool rewarding(const Transition& t) { return t.done || t.r > .5f; }  // goal reached or key collected
  void erase_pos(uint32_t idx) {
    auto it = std::find(pos_idx.begin(), pos_idx.end(), idx);
    if (it != pos_idx.end()) { *it = pos_idx.back(); pos_idx.pop_back(); }
  }
  void rebuild_pos() {
    pos_idx.clear();
    for (size_t i = 0; i < replay.size(); i++) if (rewarding(replay[i])) pos_idx.push_back((uint32_t)i);
  }
  void remember(const Transition& t) {
    size_t idx;
    const size_t cap = (size_t)cfg.rcap;
    if (replay.size() < cap) { idx = replay.size(); replay.push_back(t); if (cfg.graph > 0) pin.push_back(0); }
    else {
      if (cfg.graph > 0) {  // the ring skips pinned exemplars
        idx = ring;
        for (size_t tries = 0; tries < cap && pin[idx]; tries++) idx = (idx + 1) % cap;
        ring = (idx + 1) % cap;
      } else idx = steps % cap;
      if (rewarding(replay[idx])) erase_pos((uint32_t)idx);
      replay[idx] = t;
    }
    if (rewarding(t)) pos_idx.push_back((uint32_t)idx);
    if (cfg.graph > 0 && gr.ready && pin_count < cap / 4) {  // pin the first real transition of every graph edge
      const uint64_t key = ((uint64_t)gr.node(t.s) * ACT + (uint64_t)t.a) * 4096u + (uint64_t)gr.node(t.ns);
      if (pinmap.emplace(key, (uint32_t)idx).second) { pin[idx] = 1; pin_count++; }
    }
  }
  size_t sample_uniform() { return std::uniform_int_distribution<size_t>(0, replay.size() - 1)(rng); }
  size_t sample_idx() {
    if (cfg.pos_frac > 0.f && !pos_idx.empty() && std::uniform_real_distribution<float>(0.f, 1.f)(rng) < cfg.pos_frac)
      return pos_idx[std::uniform_int_distribution<size_t>(0, pos_idx.size() - 1)(rng)];
    return sample_uniform();
  }

  // ---- learning ----
  void learn_world() {
    double loss = 0;
    for (int m = 0; m < ENSEMBLE; m++) {  // each model draws its own bootstrap batch
      Grad g;
      for (int b = 0; b < BATCH; b++) {
        const Transition& z = replay[sample_idx()];
        Out target{};
        for (int j = 0; j < OBS; j++) target[j] = clampf((z.ns[j] - z.s[j]) / DSCALE[j], -3.f, 3.f);
        target[OBS] = clampf(z.r / REWARD_SCALE, -1.f, 1.f);
        loss += models[m].accumulate(make_in(z.s, z.c), z.a, z.c.pa, target, g);
      }
      models[m].apply(g, 1.f / BATCH, LR * lr_mult());
    }
    updates += ENSEMBLE;
    last_loss = (float)(loss / (ENSEMBLE * BATCH));
  }
  // Multi-step (open-loop) training: the model is fed its OWN predictions for K steps and every step is scored against
  // reality; gradients flow back through the whole chain (state input, velocity-change context, residual path).
  // Returns the summed loss; accumulates into *g when given. replay[i0 .. i0+K-1] must be one contiguous episode.
  static constexpr int MAXK = 8;
  double unroll_pass(const MLP& m, size_t i0, int K, Grad* g) const {
    struct Step { In x{}; std::array<float, H1> h1{}; std::array<float, H2> h2{}; Out y{}; Vec s{}, nh{}; bool clamped[OBS] = {}; int a = 0, pa = -1; };
    Step st[MAXK];
    Vec s = replay[i0].s;
    Ctx c = replay[i0].c;
    double loss = 0;
    for (int k = 0; k < K; k++) {
      const Transition& z = replay[i0 + k];
      Step& q = st[k];
      c.m = z.c.m;
      q.s = s; q.a = z.a; q.pa = c.pa;
      q.x = make_in(s, c);
      m.layers(q.x, q.a, q.pa, q.h1, q.h2, q.y);
      for (int j = 0; j < OBS; j++) {
        float v = s[j] + DSCALE[j] * q.y[j];
        q.clamped[j] = v < -1.f || v > 1.f;
        q.nh[j] = clampf(v, -1.f, 1.f);
        float e = (q.nh[j] - z.ns[j]) / DSCALE[j];
        loss += (double)e * e / OUT;
      }
      float er = q.y[OBS] - clampf(z.r / REWARD_SCALE, -1.f, 1.f);
      loss += (double)cfg.rw * er * er / OUT;
      Ctx nc;
      if (cfg.ctx) { nc.dv[0] = q.nh[2] - s[2]; nc.dv[1] = q.nh[3] - s[3]; nc.pa = z.a; }
      c = nc;
      s = q.nh;
    }
    if (!g) return loss;
    Vec gnext{}, sub{};  // d(loss)/d(input state of step k+1) from the future; pending velocity-context term for step k
    for (int k = K - 1; k >= 0; k--) {
      const Transition& z = replay[i0 + k];
      Step& q = st[k];
      Vec dtot{};
      Out dy{};
      for (int j = 0; j < OBS; j++) {
        float direct = 2.f * (q.nh[j] - z.ns[j]) / (DSCALE[j] * DSCALE[j]) / OUT;
        dtot[j] = direct + gnext[j];
        dy[j] = q.clamped[j] ? 0.f : dtot[j] * DSCALE[j];
      }
      float er = q.y[OBS] - clampf(z.r / REWARD_SCALE, -1.f, 1.f);
      dy[OBS] = 2.f * cfg.rw * er / OUT;
      In dx{};
      m.backprop(q.x, q.a, q.pa, q.h1, q.h2, dy, *g, dx);
      Vec gk{};
      for (int j = 0; j < OBS; j++) gk[j] = dx[j] + (q.clamped[j] ? 0.f : dtot[j]) + sub[j];
      sub = Vec{};
      if (cfg.ctx && k >= 1) {  // dv_k = s_k - s_{k-1} (components 2, 3)
        gk[2] += dx[OBS]; gk[3] += dx[OBS + 1];
        sub[2] -= dx[OBS]; sub[3] -= dx[OBS + 1];
      }
      gnext = gk;
    }
    return loss;
  }
  bool unroll_start_ok(size_t i0, int K) const {
    if (i0 + (size_t)K > replay.size()) return false;
    for (int k = 0; k + 1 < K; k++)
      if (replay[i0 + k].done || replay[i0 + k].ns != replay[i0 + k + 1].s) return false;
    return true;
  }
  void learn_unroll() {
    const int K = std::min(cfg.unroll, MAXK);
    for (int m = 0; m < ENSEMBLE; m++) {
      Grad g;
      int used = 0;
      for (int b = 0; b < cfg.unroll_batch; b++)
        for (int tries = 0; tries < 12; tries++) {
          size_t i0 = sample_uniform();
          if (!unroll_start_ok(i0, K)) continue;
          unroll_pass(models[m], i0, K, &g);
          used++;
          break;
        }
      if (used) models[m].apply(g, cfg.unroll_w / (float)(used * K), LR * lr_mult());
    }
  }
  void learn_rnd() {
    RndNet::Grad g;
    double err = 0;
    const int B = 16;
    for (int b = 0; b < B; b++) {
      const Vec& s = replay[sample_uniform()].ns;
      auto t = rnd_t.forward(s);
      RndNet::Act act;
      std::array<float, RO> y{}, dy{};
      rnd_p.forward(s, act, y);
      for (int k = 0; k < RO; k++) { float e = y[k] - t[k]; err += e * e / RO; dy[k] = 2.f * e / RO; }
      rnd_p.backward(s, act, dy, g);
    }
    rnd_p.apply(g, 1.f / B, LR_RND);
    float mean_err = (float)(err / B);
    rnd_ema = rnd_ready ? .99f * rnd_ema + .01f * mean_err : mean_err;
    rnd_ready = true;
  }
  // Double-DQN step on real transitions; targets are soft-updated.
  // Reward as a function of the transition's distance features (log scale resolves small distances sharply).
  static RwNet::InT rn_feat(const Vec& s, const Vec& ns) {
    auto lg = [](float x, float y) { return std::log(std::hypot(x * 2, y * 2) + .02f); };
    return {lg(s[4], s[5]), lg(s[6], s[7]), lg(ns[4], ns[5]), lg(ns[6], ns[7]), s[8], ns[8]};
  }
  float rn_reward(const Vec& s, const Vec& ns) const { return clampf(rn.forward(rn_feat(s, ns))[0], -1.f, 1.f) * REWARD_SCALE; }
  void learn_rn() {
    RwNet::Grad g;
    for (int b = 0; b < BATCH; b++) {
      const Transition& z = replay[sample_idx()];
      auto x = rn_feat(z.s, z.ns);
      RwNet::Act act;
      RwNet::OutT y{}, dy{};
      rn.forward(x, act, y);
      dy[0] = 2.f * (y[0] - clampf(z.r / REWARD_SCALE, -1.f, 1.f));
      rn.backward(x, act, dy, g);
    }
    rn.apply(g, 1.f / BATCH, LR * lr_mult());
  }
  void learn_q() {
    QNet::Grad g;
    double loss = 0;
    for (int b = 0; b < BATCH; b++) {
      const Transition& z = replay[sample_idx()];
      float tgt = z.r / QSCALE;
      if (!z.done && !z.term) {
        In nx = make_in(z.ns, next_ctx(z.s, z.ns, z.a, z.c));
        auto qo = qn.forward(nx);
        int am = (int)(std::max_element(qo.begin(), qo.end()) - qo.begin());
        tgt += cfg.gamma * qt.forward(nx)[am];
      }
      tgt = clampf(tgt, -10.f, 10.f);
      In x = make_in(z.s, z.c);
      QNet::Act act;
      std::array<float, ACT> y{}, dy{};
      qn.forward(x, act, y);
      float e = y[z.a] - tgt;
      dy[z.a] = clampf(e, -1.f, 1.f);  // Huber derivative
      loss += std::abs(e) < 1.f ? .5 * e * e : std::abs(e) - .5;
      qn.backward(x, act, dy, g);
    }
    qn.apply(g, 1.f / BATCH, LR_Q * lr_mult());
    qt.blend(qn, TAU);
    last_qloss = .98f * last_qloss + .02f * (float)(loss / BATCH);
  }
  void avg_update() {
    if (cfg.avg <= 0.f) return;
    if (!avg_ready) { avg_models = models; avg_q = qn; avg_ready = true; return; }
    for (int m = 0; m < ENSEMBLE; m++) avg_models[m].blend(models[m], cfg.avg);
    avg_q.blend(qn, cfg.avg);
  }
  // swap the averaged weights in / out (no-op without --avg); used around validation and the final evaluation
  void avg_swap() {
    if (!avg_ready) return;
    std::swap(models, avg_models); std::swap(qn, avg_q); avg_in = !avg_in;
  }
  // ---- item 2: manager ----
  static constexpr float HG = .97f;  // per-step discount inside options
  void end_option(const Vec& s1, bool done) {
    if (opt_train && opt_k > 0) {
      OptTr o; o.s0 = opt_s0; o.s1 = s1; o.g = opt_g; o.R = opt_R; o.disc = opt_disc; o.done = done;
      if (ob.size() < 20000) ob.push_back(o); else ob[hsteps % 20000] = o;
    }
    opt_g = -1;
  }
  void manage(const Vec& s, bool explore) {  // choose / keep the current option
    if (!cn.ready) { hg_on = false; return; }
    if (opt_g >= 0) {  // option NC is the null option: plan flat, without a sub-goal
      const bool reached = opt_g < cn.NC && cn.dist(s, cn.c[opt_g]) <= cn.rad[opt_g];
      if (reached || opt_left <= 0) { end_option(s, false); has_warm = false; }
    }
    if (opt_g < 0) {
      float q[520]; const int no = std::min(cn.NC, 512) + 1;
      int home = cn.nearest(s), pick = -1;
      qh.forward(s, q);
      auto allowed = [&](int k) { return k != home && (!cfg.bottleneck || k == no - 1 || bn_ok[k]); };
      if (explore && std::uniform_real_distribution<float>(0.f, 1.f)(rng) < std::max(.1f, .5f * std::exp(-(float)episodes_done / 200.f))) {
        int guard = 0;
        do { pick = (int)(rng() % (uint32_t)no); } while (!allowed(pick) && no > 1 && ++guard < 400);
        if (!allowed(pick)) pick = no - 1;
      } else { float bv = -1e30f; for (int k = 0; k < no; k++) if (allowed(k) && q[k] > bv) { bv = q[k]; pick = k; } if (pick < 0) pick = no - 1; }
      opt_g = pick; opt_s0 = s; opt_R = 0; opt_disc = 1; opt_k = 0; opt_left = cfg.hier_T; opt_train = explore;
    }
    opt_left--;
    if (opt_g < cn.NC) { hg = cn.c[opt_g]; hg_on = true; } else hg_on = false;
  }
  // ---- item 3: curiosity (error map) and item 4: graph ----
  void learn_err() {
    QNet::Grad g;
    for (int b = 0; b < BATCH; b++) {
      const Transition& z = replay[sample_idx()];
      const float e = state_error(z);
      err_ema = err_ema == 0.f ? e : .999f * err_ema + .001f * e;
      const float tgt = clampf(e / (err_ema + 1e-7f), 0.f, 5.f);
      In x = make_in(z.s, z.c);
      QNet::Act act;
      std::array<float, ACT> y{}, dy{};
      en.forward(x, act, y);
      dy[z.a] = clampf(y[z.a] - tgt, -1.f, 1.f);
      en.backward(x, act, dy, g);
    }
    en.apply(g, 1.f / BATCH, LR_Q * lr_mult());
  }
  void learn_graph(const Transition& t) {
    if (!gr.ready) gr.init(replay, cfg.graph, 7001u);
    gr.observe(t);
    if (steps % 200 == 0) gr.solve(cfg.gamma);
  }
  void learn_hier(const Transition& t) {
    if (!cn.ready) {
      cn.init(replay, cfg.hier, 9001u);
      qh.init(cn.NC + 1, 31u); qht = qh;
    }
    cn.observe(t.ns);
    if (cfg.bottleneck) { if (ep_seen.size() != (size_t)cn.NC) ep_seen.assign((size_t)cn.NC, 0); ep_seen[(size_t)cn.nearest(t.ns)] = 1; if (t.done) ep_succ = true; }
    hsteps++;
    if (ob.size() < 64) return;
    float q[520];
    for (int b = 0; b < 32; b++) {
      const OptTr& o = ob[std::uniform_int_distribution<size_t>(0, ob.size() - 1)(rng)];
      float y = o.R;
      if (!o.done) { qht.forward(o.s1, q); float m = -1e30f; for (int k = 0; k <= cn.NC; k++) m = std::max(m, q[k]); y += o.disc * m; }
      qh.forward(o.s0, q);
      std::vector<float> dq((size_t)cn.NC + 1, 0.f);
      dq[o.g] = std::max(-5.f, std::min(5.f, q[o.g] - y));
      qh.backward(o.s0, dq.data());
    }
    qh.step(LR * lr_mult(), 1.f / 32.f);
    if (hsteps % 200 == 0) qht.copy_from(qh);
  }
  void learn_tokens(const Transition& t) {
    if (!tm.ready) tm.init(replay, cfg.vq, 4242u);
    tm.observe(t.ns);
    for (int b = 0; b < BATCH; b++) tm.accumulate(replay[sample_idx()], true);
    tm.step_all(LR * lr_mult(), 1.f / BATCH);
  }
  void learn(const Transition& t) {
    if (cfg.hier > 0 && opt_g >= 0 && opt_train) { opt_R += opt_disc * t.r; opt_disc *= HG; opt_k++; last_s = t.ns; if (t.done) end_option(t.ns, true); }
    remember(t);
    steps++;
    if (replay.size() < (size_t)WARMUP) return;
    learn_world();
    if (cfg.vq > 0) learn_tokens(t);
    if (cfg.synth && steps % (cfg.synth_lib ? 5000 : 250) == 0) { sy.fit(replay); synth_prune(); }  // the formula library search is ~50x dearer than the affine one
    if (cfg.hier > 0) learn_hier(t);
    if (cfg.curious > 0.f) learn_err();
    if (cfg.graph > 0) learn_graph(t);
    if (cfg.unroll >= 2) learn_unroll();
    if (cfg.rnet) { learn_rn(); learn_rn(); }
    if (cfg.rnd_beta > 0.f) learn_rnd();
    if (cfg.value) for (int k = 0; k < cfg.q_updates; k++) learn_q();
    avg_update();
  }

  // ---- planning ----
  float novelty(const Vec& s) const {
    if (!rnd_ready) return 0.f;
    auto t = rnd_t.forward(s), y = rnd_p.forward(s);
    float e = 0;
    for (int k = 0; k < RO; k++) e += sq(y[k] - t[k]);
    return clampf(e / RO / (rnd_ema + 1e-8f), 0.f, 5.f);
  }
  float value_of(const Vec& s, const Ctx& c) const {
    auto q = qn.forward(make_in(s, c));
    return QSCALE * *std::max_element(q.begin(), q.end()) + ((cfg.graph > 0 && cfg.graph_w > 0.f && gr.ready) ? cfg.graph_w * gr.value(s) : 0.f);
  }
  float heuristic(const Vec& s) const {
    float d = std::hypot(s[4] * 2, s[5] * 2), od = std::hypot(s[6] * 2, s[7] * 2);
    return -.55f * d - .2f * std::max(0.f, .23f - od);
  }
  float rollout_score(const Vec& start, const Ctx& c0, const Seq& seq, bool explore, const Seen* mem) const {
    Vec s = start;
    Ctx c = c0;
    const bool search = mem && cfg.cover > 0.f && start[4] == 0.f && start[5] == 0.f;  // goal currently hidden
    Seen m;
    if (search) m = *mem;
    float score = 0, disc = 1;
    bool got = start[8] > 0.f;  // key-door oracle phase tracking
    const float hw = cfg.heur ? 1.f : 0.f;
    const int R = std::max(1, cfg.repeat);  // macro-actions: every planned action is held for R primitive steps
    for (int h = 0; h < cfg.horizon * R; h++) {
      const int ah = seq[h / R];
      Vec ns{};
      float r = 0, u = 0;
      predict(s, c, ah, ns, r, u);
      if (cfg.show_phase && cfg.clock_fix) { float ph = s[9] + .01f; ns[9] = ph > 1.f ? ph - 2.f : ph; }  // the clock advances deterministically
      if (cfg.rnet && !cfg.oracle) r = rn_reward(s, ns);
      if (cfg.oracle && cfg.keydoor) {  // diagnostic: true key-door reward evaluated on the predicted state
        float dk = std::hypot(ns[6] * 2, ns[7] * 2), dg = std::hypot(ns[4] * 2, ns[5] * 2);
        r = (cfg.sparse ? -.01f : -.035f - .55f * (got ? dg : dk));
        if (!got && dk < .09f) { r += 1.f; got = true; }
        else if (got && dg < .085f) r += 4.f;
      } else if (cfg.oracle) r = reward_fn(std::hypot(ns[4] * 2, ns[5] * 2), std::hypot(ns[6] * 2, ns[7] * 2), cfg.sparse).r;
      float step = r + .22f * hw * heuristic(ns);
      if (hg_on) step += cfg.hier_w * (cn.dist(s, hg) - cn.dist(ns, hg));  // progress towards the manager's concept
      if (explore) {
        if (h == 0) step += EXPLORE_BONUS * std::sqrt(u);
        if (cfg.rnd_beta > 0.f) step += cfg.rnd_beta * novelty(ns);
        if (cfg.curious > 0.f) step += cfg.curious * clampf(en.forward(make_in(s, c))[ah], 0.f, 3.f);  // go where the model is wrong
      }
      if (search) step += cfg.cover * (float)m.mark(ns[0], ns[1]);
      score += disc * step;
      c = next_ctx(s, ns, ah, c);
      s = ns;
      disc *= cfg.gamma;
    }
    if (cfg.oracle_value && cfg.keydoor) {  // diagnostic: remaining path length to key then goal, from the predicted state
      float dk = std::hypot(s[6] * 2, s[7] * 2), dg = std::hypot(s[4] * 2, s[5] * 2);
      float dkg = std::hypot((s[4] - s[6]) * 2, (s[5] - s[7]) * 2);
      score += disc * (-.35f * (got ? dg : dk + dkg));
    } else if (cfg.value) score += disc * value_of(s, c);
    else score += disc * 1.2f * hw * heuristic(s);
    return score;
  }

  int choose(const Vec& s, const Ctx& c, uint64_t ep, bool explore, const Seen* mem = nullptr) {
    std::uniform_real_distribution<float> u(0, 1);
    std::uniform_int_distribution<int> ai(0, ACT - 1);
    if (explore) {
      float eps = std::max(.03f, .30f * std::exp(-(float)ep / 150.f));
      if (replay.size() < (size_t)WARMUP || u(rng) < eps) { has_warm = false; return ai(rng); }
    }
    hg_on = false;
    if (cfg.hier > 0) manage(s, explore);
    const int H = cfg.horizon, E = std::max(3, cfg.pop / 6);
    if ((int)pop.size() != cfg.pop) pop.resize(cfg.pop);
    Probs prob;
    for (auto& row : prob) row.fill(1.f / ACT);
    if (has_warm) {  // shift the previous plan by one (macro-)step
      const int R = std::max(1, cfg.repeat);
      const bool shift = (warm_age % R) == 0;
      for (int h = 0; h < H; h++) {
        const int src = shift ? h + 1 : h;
        for (int a = 0; a < ACT; a++) prob[h][a] = src < H ? .7f * warm[src][a] + .3f / ACT : 1.f / ACT;
      }
    }
    for (int it = 0; it < cfg.iters; it++) {
      for (auto& p : pop) {
        for (int h = 0; h < H; h++) {
          float z = u(rng), acc = 0;
          int a = ACT - 1;
          for (int j = 0; j < ACT; j++) { acc += prob[h][j]; if (z <= acc) { a = j; break; } }
          p.a[h] = a;
        }
        p.score = rollout_score(s, c, p.a, explore, mem);
      }
      std::partial_sort(pop.begin(), pop.begin() + E, pop.end(), [](const Plan& a, const Plan& b) { return a.score > b.score; });
      Probs cnt{};
      for (int e = 0; e < E; e++)
        for (int h = 0; h < H; h++) cnt[h][pop[e].a[h]] += 1.f / E;
      for (int h = 0; h < H; h++) {
        float z = 0;
        for (int a = 0; a < ACT; a++) { prob[h][a] = .25f * prob[h][a] + .75f * cnt[h][a] + .002f; z += prob[h][a]; }
        for (int a = 0; a < ACT; a++) prob[h][a] /= z;
      }
    }
    warm = prob;
    has_warm = true;
    warm_age++;
    return (int)(std::max_element(prob[0].begin(), prob[0].begin() + ACT) - prob[0].begin());
  }

  // ---- diagnostics (own RNG: never disturbs training) ----
  Diag diagnostics() const {
    Diag d;
    d.pos = (int)pos_idx.size();
    if (replay.empty()) return d;
    std::mt19937 dr(777);
    int n = 0;
    for (size_t k = 0; k < pos_idx.size() && n < 200; k++, n++) {
      const Transition& t = replay[pos_idx[k]];
      Vec ns{};
      float r, u;
      predict(t.s, t.c, t.a, ns, r, u);
      d.rpos += r;
      d.qgoal += QSCALE * qn.forward(make_in(t.s, t.c))[t.a];
    }
    if (n) { d.rpos /= n; d.qgoal /= n; }
    int m = 0, q = 0;
    for (int k = 0; k < 600 && (m < 300 || q < 300); k++) {
      const Transition& t = replay[dr() % replay.size()];
      if (m < 300 && !t.done) {
        Vec ns{};
        float r, u;
        predict(t.s, t.c, t.a, ns, r, u);
        d.rneg += r; m++;
      }
      if (q < 300) { d.qmean += value_of(t.s, t.c); q++; }
    }
    if (m) d.rneg /= m;
    if (q) d.qmean /= q;
    return d;
  }

  // ---- checkpoint: atomic write, strict validation, nothing changes unless the whole file is valid ----
  bool save(const std::string& path) const {
    std::string tmp = path + ".tmp";
    {
      std::ofstream o(tmp, std::ios::binary);
      if (!o) return false;
      const char magic[8] = {'N', 'X', 'C', 'O', 'G', '5', '\0', '\0'};
      o.write(magic, 8);
      uint32_t version = 5, dims[8] = {OBS, ACT, H1, H2, ENSEMBLE, RH, QH, MEM};
      wr(o, version);
      o.write(reinterpret_cast<const char*>(dims), sizeof(dims));
      wr(o, steps); wr(o, updates); wr(o, episodes_done); wr(o, goals);
      wr(o, last_loss); wr(o, last_qloss); wr(o, rnd_ema); wr(o, (uint8_t)rnd_ready);
      for (auto& m : models) m.save(o);
      rnd_t.save(o); rnd_p.save(o); qn.save(o); qt.save(o);
      if (cfg.rnet) rn.save(o);
      uint64_t n = replay.size();
      wr(o, n);
      for (auto& t : replay) {
        o.write(reinterpret_cast<const char*>(t.s.data()), sizeof(t.s));
        o.write(reinterpret_cast<const char*>(t.ns.data()), sizeof(t.ns));
        int32_t a = t.a, pa = t.c.pa;
        uint8_t d = (uint8_t)((t.done ? 1 : 0) | (t.term ? 2 : 0));
        wr(o, a); wr(o, t.r); wr(o, d); wr(o, t.task);
        wr(o, t.c.dv[0]); wr(o, t.c.dv[1]); wr(o, pa); put(o, t.c.m);
      }
      if (cfg.vq > 0) tm.save(o);
      if (cfg.hier > 0) { cn.save(o); if (cn.ready) { qh.save(o); qht.save(o); } }
      if (cfg.bottleneck) { put(o, bn_s); put(o, bn_f); wr(o, bn_ns); wr(o, bn_nf); }
      if (cfg.curious > 0.f) { en.save(o); wr(o, err_ema); }
      if (cfg.graph > 0) {
        gr.save(o); wr(o, (uint64_t)ring);
        uint64_t np = pinmap.size(); wr(o, np);
        for (const auto& kv : pinmap) { wr(o, kv.first); wr(o, kv.second); }
      }
      o.flush();
      if (!o.good()) return false;
    }
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    if (ec) { std::filesystem::remove(path, ec); ec.clear(); std::filesystem::rename(tmp, path, ec); }
    return !ec;
  }
  bool load(const std::string& path) {
    std::ifstream i(path, std::ios::binary);
    if (!i) return false;
    char magic[8]{};
    uint32_t version = 0, dims[8]{};
    i.read(magic, 8);
    rd(i, version);
    i.read(reinterpret_cast<char*>(dims), sizeof(dims));
    if (!i || std::memcmp(magic, "NXCOG5", 6) != 0 || version != 5) return false;
    if (dims[0] != OBS || dims[1] != ACT || dims[2] != H1 || dims[3] != H2 || dims[4] != ENSEMBLE || dims[5] != RH || dims[6] != QH || dims[7] != MEM) return false;
    uint64_t st = 0, up = 0, ed = 0, gl = 0;
    float ll = 0, lq = 0, re = 0;
    uint8_t rr = 0;
    rd(i, st); rd(i, up); rd(i, ed); rd(i, gl); rd(i, ll); rd(i, lq); rd(i, re); rd(i, rr);
    if (!i || !std::isfinite(ll) || !std::isfinite(lq) || !std::isfinite(re)) return false;
    auto tm = models;
    auto trt = rnd_t, trp = rnd_p;
    auto tqn = qn, tqt = qt;
    for (auto& m : tm) if (!m.load(i)) return false;
    auto trn = rn;
    if (!trt.load(i) || !trp.load(i) || !tqn.load(i) || !tqt.load(i)) return false;
    if (cfg.rnet && !trn.load(i)) return false;
    uint64_t n = 0;
    rd(i, n);
    if (!i || n > (uint64_t)REPLAY_CAP) return false;
    std::vector<Transition> tmp;
    tmp.reserve((size_t)n);
    for (uint64_t k = 0; k < n; k++) {
      Transition t;
      int32_t a = 0, pa = -1;
      uint8_t d = 0;
      i.read(reinterpret_cast<char*>(t.s.data()), sizeof(t.s));
      i.read(reinterpret_cast<char*>(t.ns.data()), sizeof(t.ns));
      rd(i, a); rd(i, t.r); rd(i, d); rd(i, t.task);
      rd(i, t.c.dv[0]); rd(i, t.c.dv[1]); rd(i, pa); get(i, t.c.m);
      if (!i || a < 0 || a >= ACT || pa < -1 || pa >= ACT || !std::isfinite(t.r) || !std::isfinite(t.c.m[0])) return false;
      t.a = a; t.c.pa = pa; t.done = (d & 1) != 0; t.term = (d & 2) != 0;
      tmp.push_back(t);
    }
    TokenModel ttm;
    if (cfg.vq > 0 && !ttm.load(i)) return false;
    Concepts tcn; QHi tqh, tqht;
    QNet ten = en; float terr = 0.f;
    Graph tgr; uint64_t tring = 0; std::unordered_map<uint64_t, uint32_t> tpin;
    if (cfg.hier > 0) {
      if (!tcn.load(i)) return false;
      if (tcn.ready) { tqh.init(tcn.NC + 1, 31u); tqht = tqh; if (!tqh.load(i) || !tqht.load(i)) return false; }
    }
    std::array<float, 512> tbs{}, tbf{}; float tbns = 0, tbnf = 0;
    if (cfg.bottleneck) { get(i, tbs); get(i, tbf); rd(i, tbns); rd(i, tbnf); if (!i || !std::isfinite(tbns) || !std::isfinite(tbnf)) return false; }
    if (cfg.curious > 0.f) { if (!ten.load(i)) return false; rd(i, terr); if (!i || !std::isfinite(terr)) return false; }
    if (cfg.graph > 0) {
      if (!tgr.load(i)) return false;
      uint64_t np = 0; rd(i, tring); rd(i, np);
      if (!i || np > n) return false;
      for (uint64_t k = 0; k < np; k++) { uint64_t key = 0; uint32_t idx = 0; rd(i, key); rd(i, idx); if (!i || idx >= n) return false; tpin[key] = idx; }
    }
    rn = trn; models = tm; rnd_t = trt; rnd_p = trp; qn = tqn; qt = tqt;
    if (cfg.vq > 0) this->tm = ttm;
    if (cfg.hier > 0) { cn = tcn; qh = tqh; qht = tqht; ob.clear(); opt_g = -1; hsteps = 0; }
    if (cfg.bottleneck) { bn_s = tbs; bn_f = tbf; bn_ns = tbns; bn_nf = tbnf; bn_ok.fill(0); ep_seen.clear(); ep_succ = false; bn_ns += 0; if (cn.ready) { const float s0 = bn_ns, f0 = bn_nf; bn_finish_episode_stats_only(s0, f0); } }
    if (cfg.curious > 0.f) { en = ten; err_ema = terr; }
    if (cfg.graph > 0) {
      gr = tgr; ring = (size_t)tring; pinmap = tpin; pin.assign((size_t)n, 0); pin_count = 0;
      for (const auto& kv : pinmap) if (!pin[kv.second]) { pin[kv.second] = 1; pin_count++; }
    }
    replay.swap(tmp);
    steps = st; updates = up; episodes_done = ed; goals = gl;
    last_loss = ll; last_qloss = lq; rnd_ema = re; rnd_ready = rr != 0;
    rebuild_pos();
    return true;
  }
};

// ---------------------------------------------------------------------------------------------
// Episodes, evaluation, reference policies
// ---------------------------------------------------------------------------------------------
struct Stats { double success = 0, reward = 0, steps = 0, model_err = 0, persist_err = 0, found = 0; };  // found: goal appeared in the observation at least once

Stats run_episode(Agent& ag, World& w, const Scn& sc, uint64_t ep, bool train) {
  Vec s = w.reset(sc);
  Stats st;
  Ctx c;
  Seen mem;
  mem.mark(s[0], s[1]);
  if (!w.keydoor && (s[4] != 0.f || s[5] != 0.f)) st.found = 1;  // key-door: 'found' means the key was collected
  GoalMem gm;
  if (cfg.gmem) s = gm.fix(s);
  ag.begin_episode();
  for (int t = 0; t < w.maxT(); t++) {
    int a = ag.choose(s, c, ep, train, cfg.cover > 0.f ? &mem : nullptr);
    Transition tr = w.step(a);
    tr.c = c;
    mem.mark(tr.ns[0], tr.ns[1]);
    if (w.keydoor ? w.has_key : (tr.ns[4] != 0.f || tr.ns[5] != 0.f)) st.found = 1;
    if (cfg.gmem) { tr.s = s; tr.ns = gm.fix(tr.ns); }  // the agent learns from its memory-completed observations
    st.reward += tr.r; st.steps++;
    if (train) { if (cfg.blind) { Transition tb = tr; tb.r = 0.f; tb.done = false; ag.learn(tb); } else ag.learn(tr); }
    else {  // held-out model quality, never trained on
      st.model_err += ag.state_error(tr);
      float pe = 0;
      for (int j = 0; j < OBS; j++) pe += sq(tr.ns[j] - tr.s[j]);
      st.persist_err += pe / OBS;  // trivial baseline: "nothing changes"
    }
    c = next_ctx(tr.s, tr.ns, a, c);
    s = tr.ns;
    if (tr.done) { st.success = 1; break; }
  }
  return st;
}

int greedy_action(const World& w) {  // one-step lookahead on the TRUE simulator with dense reward (cheating reference)
  if (w.game == 100) return 0;  // no reference policy for an external environment (the oracle column is meaningless there)
  if (w.game >= 10) return Puzzle::dist_first(w.pz.nx, w.pz_s, w.pz_g);  // held-out family: exact BFS optimum
  if (w.game == 2) {  // exact shortest chain by breadth-first search over 0..255 (the optimum the agent is compared with)
    auto apply = [](int v, int a) { int nv = v; switch (a) { case 0: nv = v + 1; break; case 1: nv = v + 2; break; case 2: nv = v + 4; break; case 3: nv = v + 8; break; case 4: nv = v * 2; break; case 5: nv = v * 3; break; case 6: nv = v - 1; break; case 7: nv = v - 4; break; default: nv = v % 2 == 0 ? v / 2 : v; } return (nv >= 0 && nv <= 255) ? nv : v; };
    std::array<int, 256> first; first.fill(-1); std::array<int, 256> q{}; int qh = 0, qt = 0; first[w.mv] = 9; q[qt++] = w.mv;
    while (qh < qt) { int v = q[qh++]; if (v == w.mt) break; for (int a = 0; a < 9; a++) { int nv = apply(v, a); if (first[nv] < 0) { first[nv] = v == w.mv ? a : first[v]; q[qt++] = nv; } } }
    return first[w.mt] >= 0 && first[w.mt] < 9 ? first[w.mt] : 0;
  }
  if (w.game == 3) {  // Pendulum reference: energy pumping far from the top, PD controller near it
    const float E = .5f * w.pw * w.pw + 15.f * std::cos(w.pth);
    float u;
    if (std::fabs(w.pth) < .5f && E > 13.f) u = clampf(-(15.f * w.pth + 6.f * w.pw) / 3.f, -2.f, 2.f);
    else u = (E < 15.f ? 1.f : -1.f) * (w.pw >= 0 ? 2.f : -2.f);
    return u > .5f ? 1 : u < -.5f ? 5 : 0;
  }
  if (w.game == 4) {  // exact shortest press sequence by BFS over the 256 states
    std::array<int, 256> first; first.fill(-1); std::array<int, 256> q{}; int qh = 0, qt = 0; first[w.sw] = 9; q[qt++] = w.sw;
    while (qh < qt) { int v = q[qh++]; if (v == 255) break; for (int pr = 0; pr < 8; pr++) { int nv = v; for (int k = std::max(0, pr - 1); k <= std::min(7, pr + 1); k++) nv ^= 1 << k; if (first[nv] < 0) { first[nv] = v == w.sw ? pr : first[v]; q[qt++] = nv; } } }
    return first[255] >= 0 && first[255] < 9 ? first[255] : 0;
  }
  if (w.game == 5) {  // exact shortest chain by BFS over 0..96
    auto apply = [](int v, int a) { switch (a) { case 0: return (v + 1) % 97; case 1: return (v + 5) % 97; case 2: return (v + 17) % 97; case 3: return (v * 2) % 97; case 4: return (v * 3) % 97; case 5: return (v + 96) % 97; case 6: return (v + 90) % 97; case 7: return (v * 5) % 97; default: return (v * 2 + 1) % 97; } };
    std::array<int, 97> first; first.fill(-1); std::array<int, 97> q{}; int qh = 0, qt = 0; first[w.mv] = 9; q[qt++] = w.mv;
    while (qh < qt) { int v = q[qh++]; if (v == w.mt) break; for (int a = 0; a < 9; a++) { int nv = apply(v, a); if (first[nv] < 0) { first[nv] = v == w.mv ? a : first[v]; q[qt++] = nv; } } }
    return first[w.mt] >= 0 && first[w.mt] < 9 ? first[w.mt] : 0;
  }
  if (w.game == 1) return w.mcv >= 0 ? 1 : 5;  // MountainCar reference: push in the direction of the velocity (energy pumping)
  int best = 0;
  float bs = -1e9f;
  for (int a = 0; a < ACT; a++) {
    World c = w;
    c.sparse = false;
    c.force_dense = true;  // reference policy may cheat: it sees the dense reward even in hidden-goal tasks
    float r = c.step(a).r;
    if (r > bs) { bs = r; best = a; }
  }
  return best;
}
Stats run_baseline(World& w, const Scn& sc, bool greedy, std::mt19937& rng) {
  std::uniform_int_distribution<int> ai(0, ACT - 1);
  w.reset(sc);
  Stats st;
  for (int t = 0; t < w.maxT(); t++) {
    Transition tr = w.step(greedy ? greedy_action(w) : ai(rng));
    st.reward += tr.r; st.steps++;
    if (tr.done) { st.success = 1; break; }
  }
  return st;
}

struct ScnResult { double succ = 0, rew = 0, steps = 0, merr = 0, perr = 0, found = 0, succ_found = 0; };
// Frozen-agent evaluation on identical, seed-determined episodes. Does not change any training state.
ScnResult eval_scenario(Agent& ag, uint32_t seed, int id, int n, uint64_t ep) {
  auto saved = ag.rng;
  World ev(seed + 9999 + (uint32_t)id * 1000);
  Scn sc = scenario(id);
  ScnResult r;
  double nst = 0, nfound = 0, nsf = 0;
  for (int j = 0; j < n; j++) {
    Stats q = run_episode(ag, ev, sc, ep, false);
    nfound += q.found; nsf += (q.found > 0 && q.success > 0) ? 1 : 0;
    r.succ += q.success; r.rew += q.reward; r.steps += q.steps; r.merr += q.model_err; r.perr += q.persist_err; nst += q.steps;
  }
  ag.rng = saved;
  r.succ /= n; r.rew /= n; r.steps /= n;
  r.found = nfound / n; r.succ_found = nfound > 0 ? nsf / nfound : 0;
  r.merr /= std::max(1.0, nst); r.perr /= std::max(1.0, nst);
  return r;
}

// Open-loop prediction error: roll the model forward with the actions the agent really took and compare the position
// (obs 0,1; units of the arena) with the true one, k = 1..8 steps ahead, over frozen episodes.
void multistep_diag(Agent& ag, uint32_t seed, int id, int n) {
  auto saved = ag.rng;
  World ev(seed + 9999 + (uint32_t)id * 1000);
  Scn sc = scenario(id);
  constexpr int K = 8;
  double err[K + 1] = {}, cnt[K + 1] = {};
  for (int j = 0; j < n; j++) {
    Vec s = ev.reset(sc);
    Ctx c;
    ag.begin_episode();
    std::vector<Vec> S{s};
    std::vector<Ctx> C{c};
    std::vector<int> A;
    for (int t = 0; t < MAX_STEPS; t++) {
      int a = ag.choose(s, c, ag.episodes_done, false, nullptr);
      Transition tr = ev.step(a);
      A.push_back(a);
      c = next_ctx(tr.s, tr.ns, a, c);
      s = tr.ns;
      S.push_back(s); C.push_back(c);
      if (tr.done) break;
    }
    for (size_t t0 = 0; t0 + 1 < S.size(); t0 += 4) {
      Vec ps = S[t0];
      Ctx pc = C[t0];
      for (int k = 1; k <= K && t0 + k < S.size(); k++) {
        Vec ns{};
        float r, u;
        ag.predict(ps, pc, A[t0 + k - 1], ns, r, u);
        pc = next_ctx(ps, ns, A[t0 + k - 1], pc);
        ps = ns;
        err[k] += std::hypot(ps[0] - S[t0 + k][0], ps[1] - S[t0 + k][1]);
        cnt[k] += 1;
      }
    }
  }
  ag.rng = saved;
  std::cout << "  " << std::left << std::setw(11) << scenario(id).name << std::right << " pos error after k steps (arena units): ";
  for (int k : {1, 2, 4, 6, 8}) std::cout << "k" << k << "=" << std::setprecision(4) << err[k] / std::max(1.0, cnt[k]) << " ";
  std::cout << "\n";
}

// Behaviour trace: where do failing episodes get stuck?
void trace_diag(Agent& ag, uint32_t seed, int id, int n) {
  auto saved = ag.rng;
  World ev(seed + 9999 + (uint32_t)id * 1000);
  Scn sc = scenario(id);
  std::cout << "  " << scenario(id).name << "  (ep: result | closest approach to key | closest to goal after key | mean speed | frac. steps at wall | steps)\n";
  for (int j = 0; j < n; j++) {
    Vec s = ev.reset(sc);
    Ctx c;
    ag.begin_episode();
    float mink = 9, ming = 9, sp = 0;
    int wall = 0, t = 0;
    bool ok = false;
    for (; t < MAX_STEPS; t++) {
      int a = ag.choose(s, c, ag.episodes_done, false, nullptr);
      Transition tr = ev.step(a);
      c = next_ctx(tr.s, tr.ns, a, c);
      s = tr.ns;
      if (!ev.has_key) mink = std::min(mink, std::hypot(ev.kx - ev.x, ev.ky - ev.y));
      else ming = std::min(ming, std::hypot(ev.gx - ev.x, ev.gy - ev.y));
      sp += std::hypot(ev.vx, ev.vy);
      if (std::abs(ev.x) > .95f || std::abs(ev.y) > .95f) wall++;
      if (tr.done) { ok = true; t++; break; }
    }
    std::cout << "    " << std::setw(2) << j << ": " << (ok ? "OK  " : ev.has_key ? "key " : "FAIL") << std::fixed << std::setprecision(2) << " | " << mink
              << " | " << (ev.has_key ? ming : -1.f) << " | " << sp / std::max(1, t) << " | " << (float)wall / std::max(1, t) << " | " << t << "\n";
  }
  ag.rng = saved;
}

std::vector<int> eval_ids();
// Accuracy of the next-token model on fresh episodes of the evaluation scenarios (per active dimension), exact-reward detection included.
void tok_diag(Agent& ag, uint32_t seed, int id, int n) {
  if (!ag.tm.ready) { std::cout << "token model not ready\n"; return; }
  World ev(seed + 777 + (uint32_t)id * 1000);
  Scn sc = scenario(id);
  const int na = ag.tm.na();
  std::vector<double> ok(na, 0), tot(na, 0), okm(na, 0), totm(na, 0);
  double rpos_ok = 0, rpos = 0, rfalse = 0, rneg = 0;
  for (int e = 0; e < n; e++) {
    Vec s = ev.reset(sc); Ctx c; ag.begin_episode();
    for (int t = 0; t < ev.maxT(); t++) {
      const int a = (int)(ag.rng() % ACT);
      Transition tr = ev.step(a);
      Vec ns; float r;
      ag.tm.predict(s, a, ns, r);
      int tn[OBS]; ag.tm.toks(tr.ns, tn); int ts[OBS]; ag.tm.toks(s, ts);
      for (int i = 0; i < na; i++) {
        const int j = ag.tm.ad[i]; const bool good = ag.tm.tok(j, ns[j]) == tn[i];
        tot[i]++; ok[i] += good;
        if (tn[i] != ts[i]) { totm[i]++; okm[i] += good; }  // only transitions where the token really changes
      }
      if (tr.r > 1.f) { rpos++; rpos_ok += r > 1.f; } else { rneg++; rfalse += r > 1.f; }
      s = tr.ns; if (tr.done) break;
    }
  }
  std::cout << "token model accuracy on fresh random-action episodes (scenario " << sc.name << "):\n";
  for (int i = 0; i < na; i++) std::cout << "  dim " << ag.tm.ad[i] << ": all " << std::fixed << std::setprecision(1) << 100 * ok[i] / std::max(1.0, tot[i]) << "%   when the token changes " << 100 * okm[i] / std::max(1.0, totm[i]) << "% (" << (long)totm[i] << " cases)\n";
  std::cout << "  goal reward detected: " << 100 * rpos_ok / std::max(1.0, rpos) << "% of " << (long)rpos << "; false goal alarms: " << rfalse << " of " << (long)rneg << " other steps\n";
}
// Reward-head check on fresh random-action episodes: mean predicted reward on failure steps (term), goal steps and ordinary steps.
void rw_diag(Agent& ag, uint32_t seed, int id, int n) {
  World ev(seed + 555 + (uint32_t)id * 1000);
  Scn sc = scenario(id);
  double sum[3] = {}, cnt[3] = {}, hit[3] = {};  // 0 ordinary, 1 failure, 2 goal
  for (int e = 0; e < n; e++) {
    Vec s = ev.reset(sc); Ctx c; ag.begin_episode();
    for (int t = 0; t < ev.maxT(); t++) {
      const int a = (int)(ag.rng() % ACT);
      Transition tr = ev.step(a);
      Vec ns; float r, u; ag.predict(s, c, a, ns, r, u);
      const int k = tr.term ? 1 : tr.done ? 2 : 0;
      sum[k] += r; cnt[k]++;
      hit[k] += k == 1 ? (r < -.3f) : k == 2 ? (r > 1.f) : (std::abs(r) < .3f);
      c = next_ctx(tr.s, tr.ns, a, c); s = tr.ns; if (tr.done) break;
    }
  }
  static const char* const nm[3] = {"ordinary steps", "failure steps", "goal steps"};
  std::cout << "reward head on fresh random-action episodes (scenario " << sc.name << "):\n";
  for (int k = 0; k < 3; k++) if (cnt[k] > 0) std::cout << "  " << nm[k] << ": " << (long)cnt[k] << ", mean predicted reward " << std::fixed << std::setprecision(3) << sum[k] / cnt[k] << ", recognised " << std::setprecision(1) << 100 * hit[k] / cnt[k] << "%\n";
}
// Validation score used for best-snapshot selection: mean success over the final-evaluation scenarios on episodes that are
// FIXED (same starts at every call) and different from the final-evaluation ones (seed offset), so selection never sees the test.
double validate(Agent& ag, uint32_t seed, int n, uint64_t ep) {
  const std::vector<int> ids = cfg.val_ids.empty() ? eval_ids() : cfg.val_ids;
  double s = 0;
  const bool sw = !ag.avg_in; if (sw) ag.avg_swap();
  for (int id : ids) s += eval_scenario(ag, seed + 45000u, id, n, ep).succ;
  if (sw) ag.avg_swap();
  return s / (double)ids.size();
}

std::vector<int> eval_ids() {
  std::vector<int> ids;
  if (!cfg.final_ids.empty()) ids = cfg.final_ids;
  else if (cfg.final_hidden) ids = {9, 10, 11};
  else if (cfg.final_keydoor) ids = {12, 13, 14};
  else for (int i = 0; i < (cfg.suite ? N_SCN : N_SEEN); i++) ids.push_back(i);
  return ids;
}

std::string cfg_string() {
  std::ostringstream o;
  o << (cfg.sparse ? "sparse" : "dense") << " heur=" << cfg.heur << " value=" << cfg.value << " H=" << cfg.horizon
    << " pop=" << cfg.pop << "x" << cfg.iters << " rnd=" << cfg.rnd_beta << " cover=" << cfg.cover << " posf=" << cfg.pos_frac
    << " gamma=" << cfg.gamma << " ctx=" << cfg.ctx << " label=" << cfg.label << " dr=" << cfg.randomize << " hidden=" << cfg.hidden << " unroll=" << cfg.unroll << " repeat=" << cfg.repeat << " showphase=" << cfg.show_phase << " showwind=" << cfg.show_wind << " rnet=" << cfg.rnet << " rw=" << cfg.rw << " keydoor=" << cfg.keydoor << " gmem=" << cfg.gmem << " res=" << cfg.reservoir
    << " oracle=" << cfg.oracle << " vq=" << cfg.vq << " hier=" << cfg.hier << " curious=" << cfg.curious << " graph=" << cfg.graph << " blind=" << cfg.blind << " synth=" << cfg.synth << " bn=" << cfg.bottleneck << " lib=" << cfg.synth_lib;
  return o.str();
}

// ---------------------------------------------------------------------------------------------
// Self tests
// ---------------------------------------------------------------------------------------------
int selftest() {
  int fails = 0, checks = 0;
  auto CHECK = [&](bool ok, const std::string& what) {
    checks++;
    if (!ok) { fails++; std::cout << "  FAIL: " << what << "\n"; }
    else std::cout << "  ok:   " << what << "\n";
  };
  Cfg saved = cfg;
  std::mt19937 R(2024);
  std::uniform_real_distribution<float> U(-1.f, 1.f);

  std::cout << "[1] fast tanh\n";
  {
    double maxerr = 0, prev = -2;
    bool mono = true, bounded = true;
    for (int i = -120000; i <= 120000; i++) {
      float x = (float)i * 1e-4f, a = ftanh(x);
      maxerr = std::max(maxerr, std::abs((double)a - std::tanh((double)x)));
      if (a < prev - 1e-6) mono = false;
      if (std::abs(a) > 1.000001f || !std::isfinite(a)) bounded = false;
      prev = a;
    }
    CHECK(maxerr < 2e-6, "max abs error vs std::tanh = " + std::to_string(maxerr));
    CHECK(mono && bounded, "monotone and bounded in [-1,1]");
  }

  auto gradcheck = [&](const char* name, float* p, const float* g, size_t n, auto&& lossfn, int samples) {
    int bad = 0;
    double worst = 0;
    for (int k = 0; k < samples; k++) {
      size_t i = R() % n;
      float old = p[i];
      const float eps = 2e-3f;
      p[i] = old + eps; double l1 = lossfn();
      p[i] = old - eps; double l2 = lossfn();
      p[i] = old;
      double num = (l1 - l2) / (2.0 * eps), an = g[i];
      double err = std::abs(num - an);
      worst = std::max(worst, err);
      if (err > 3e-4 + 0.03 * std::abs(an)) bad++;
    }
    CHECK(bad == 0, std::string("gradient check ") + name + " (worst abs diff " + std::to_string(worst) + ")");
  };

  std::cout << "[2] gradients (analytic vs finite differences)\n";
  for (int pa : {-1, 4}) {
    MLP net(5);
    for (auto& x : net.w3) x *= 2.f;
    In x{};
    for (auto& v : x) v = U(R);
    Out t{};
    for (auto& v : t) v = U(R);
    const int a = 3;
    Grad g;
    net.accumulate(x, a, pa, t, g);
    auto lossfn = [&]() { Grad tmp; return (double)net.accumulate(x, a, pa, t, tmp); };
    std::string tag = pa < 0 ? "MLP(no prev action)" : "MLP(prev action)";
    gradcheck((tag + " w1").c_str(), net.w1.data(), g.w1.data(), net.w1.size(), lossfn, 40);
    gradcheck((tag + " b1").c_str(), net.b1.data(), g.b1.data(), net.b1.size(), lossfn, 15);
    gradcheck((tag + " w2").c_str(), net.w2.data(), g.w2.data(), net.w2.size(), lossfn, 40);
    gradcheck((tag + " b2").c_str(), net.b2.data(), g.b2.data(), net.b2.size(), lossfn, 15);
    gradcheck((tag + " w3").c_str(), net.w3.data(), g.w3.data(), net.w3.size(), lossfn, 40);
    gradcheck((tag + " b3").c_str(), net.b3.data(), g.b3.data(), net.b3.size(), lossfn, 11);
    // the one-hot columns of the first layer must receive gradient
    int bad = 0;
    for (int j = 0; j < H1; j += 7) {
      size_t i1 = (size_t)j * IN + CIN + a, i2 = (size_t)j * IN + CIN + ACT + (pa < 0 ? 0 : pa);
      float old = net.w1[i1], eps = 2e-3f;
      net.w1[i1] = old + eps; double l1 = lossfn();
      net.w1[i1] = old - eps; double l2 = lossfn();
      net.w1[i1] = old;
      if (std::abs((l1 - l2) / (2 * eps) - g.w1[i1]) > 3e-4 + .03 * std::abs(g.w1[i1])) bad++;
      if (pa < 0 && g.w1[i2] != 0.f) bad++;  // previous-action column must stay untouched when absent
    }
    CHECK(bad == 0, tag + " one-hot columns");
  }
  {
    DenseNet<OBS, 12, 10, RO> net;  // small shape = same code path, faster checks
    net.init(7, 2.f);
    DenseNet<OBS, 12, 10, RO>::InT x{};
    DenseNet<OBS, 12, 10, RO>::OutT t{};
    for (auto& v : x) v = U(R);
    for (auto& v : t) v = U(R);
    DenseNet<OBS, 12, 10, RO>::Grad g;
    DenseNet<OBS, 12, 10, RO>::Act act;
    DenseNet<OBS, 12, 10, RO>::OutT y{}, dy{};
    net.forward(x, act, y);
    for (int k = 0; k < RO; k++) dy[k] = y[k] - t[k];
    net.backward(x, act, dy, g);
    auto lossfn = [&]() { auto yy = net.forward(x); double l = 0; for (int k = 0; k < RO; k++) l += .5 * sq(yy[k] - t[k]); return l; };
    gradcheck("DenseNet w1", net.w1.data(), g.w1.data(), net.w1.size(), lossfn, 40);
    gradcheck("DenseNet b1", net.b1.data(), g.b1.data(), net.b1.size(), lossfn, 10);
    gradcheck("DenseNet w2", net.w2.data(), g.w2.data(), net.w2.size(), lossfn, 40);
    gradcheck("DenseNet b2", net.b2.data(), g.b2.data(), net.b2.size(), lossfn, 10);
    gradcheck("DenseNet w3", net.w3.data(), g.w3.data(), net.w3.size(), lossfn, 40);
    gradcheck("DenseNet b3", net.b3.data(), g.b3.data(), net.b3.size(), lossfn, 10);
  }

  std::cout << "[3] environment\n";
  {
    bool ok_start = true, ok_bounds = true, ok_obst = true;
    for (int id = 0; id < N_SCN; id++) {
      World w(100 + id);
      Scn sc = scenario(id);
      for (int k = 0; k < 300; k++) {
        w.reset(sc);
        if (std::hypot(w.gx - w.x, w.gy - w.y) < .6f) ok_start = false;
        if (std::abs(w.x) > .82f || std::abs(w.y) > .82f || std::abs(w.gx) > .82f || std::abs(w.gy) > .82f) ok_bounds = false;
        if (std::hypot(w.ox - w.x, w.oy - w.y) < .3f - 1e-4f || std::hypot(w.ox - w.gx, w.oy - w.gy) < .22f - 1e-4f) ok_obst = false;
      }
    }
    CHECK(ok_start, "goal never starts within 0.6 of the agent (all scenarios)");
    CHECK(ok_bounds, "agent/goal start inside the arena (all scenarios)");
    CHECK(ok_obst, "obstacle never spawns on the agent or on the goal (all scenarios)");
    World w(5);
    w.reset(scenario(0));
    w.x = w.gx; w.y = w.gy; w.vx = w.vy = 0; w.windx = w.windy = 0; w.ox = -2.f; w.oy = -2.f;
    w.sparse = false;
    Transition tr = w.step(0);
    CHECK(tr.done && std::abs(tr.r - 3.965f) < 1e-3f, "dense reward at goal = 3.965 (got " + std::to_string(tr.r) + ")");
    w.reset(scenario(0));
    w.x = w.gx; w.y = w.gy; w.vx = w.vy = 0; w.windx = w.windy = 0; w.ox = -2.f; w.oy = -2.f;
    w.sparse = true;
    tr = w.step(0);
    CHECK(tr.done && std::abs(tr.r - 3.99f) < 1e-3f, "sparse reward at goal = 3.99 (got " + std::to_string(tr.r) + ")");
    Reward rw = reward_fn(.5f, .05f, false);
    CHECK(rw.hit && !rw.done, "obstacle hit is detected");
    World a(11), b(11);
    a.reset(scenario(4)); b.reset(scenario(4));
    bool same = true;
    for (int k = 0; k < 50; k++) { auto ta = a.step(k % ACT), tb = b.step(k % ACT); if (ta.ns != tb.ns || ta.r != tb.r) same = false; }
    CHECK(same, "environment is deterministic for a given seed");
    // hidden-goal task: goal vector masked beyond the sense radius, revealed inside it, reward forced sparse
    World wh(9);
    wh.reset(scenario(9));
    wh.x = 0; wh.y = 0; wh.gx = .8f; wh.gy = 0; wh.vx = wh.vy = 0;
    Vec far_o = wh.obs();
    wh.gx = .3f;
    Vec near_o = wh.obs();
    CHECK(far_o[4] == 0.f && far_o[5] == 0.f && far_o[8] == -1.f && near_o[4] > 0.f && near_o[8] == 1.f, "hidden goal: masked when far, visible when near");
    wh.x = wh.gx; wh.y = wh.gy; wh.windx = wh.windy = 0; wh.ox = -2.f; wh.oy = -2.f;
    CHECK(std::abs(wh.step(0).r - 3.99f) < 1e-3f, "hidden-goal task uses the sparse reward even when --sparse is off");
    GoalMem gmt;
    Vec o1{}, o2{};
    o1[0] = 0.f; o1[1] = 0.f; o1[4] = .15f; o1[5] = 0.f;  // goal 0.3 to the right, visible
    o2[0] = .1f; o2[1] = 0.f;                              // agent moved 0.1 right, goal now masked
    Vec f1 = gmt.fix(o1), f2 = gmt.fix(o2);
    CHECK(f1 == o1 && std::abs(f2[4] - .1f) < 1e-6f && f2[5] == 0.f, "goal memory completes the masked goal vector by dead reckoning");
    GoalMem gm0;
    CHECK(gm0.fix(o2) == o2, "goal memory never invents a goal before one was seen");
    Seen sm;
    int n1 = sm.mark(0.f, 0.f), n2 = sm.mark(0.f, 0.f), n3 = sm.mark(.9f, .9f);
    CHECK(n1 > 6 && n1 < 20 && n2 == 0 && n3 > 0 && sm.count() == n1 + n3, "search memory: new cells counted once, revisits add nothing");
    World wk(21);
    wk.reset(scenario(12));
    wk.sparse = true; wk.windx = wk.windy = 0; wk.vx = wk.vy = 0;
    wk.x = wk.gx; wk.y = wk.gy;
    Transition k1 = wk.step(0);
    CHECK(!k1.done && !wk.has_key && k1.s[8] == -1.f, "key-door: the goal is inert without the key");
    wk.x = wk.kx; wk.y = wk.ky; wk.vx = wk.vy = 0;
    Transition k2 = wk.step(0);
    CHECK(wk.has_key && std::abs(k2.r - .99f) < 1e-3f && k2.ns[6] == 0.f && k2.ns[7] == 0.f && k2.ns[8] == 1.f && k2.s[8] == -1.f, "key-door: key pickup gives +1, the key vector vanishes and the phase flag flips");
    wk.x = wk.gx; wk.y = wk.gy; wk.vx = wk.vy = 0;
    Transition k3 = wk.step(0);
    CHECK(k3.done && std::abs(k3.r - 3.99f) < 1e-3f, "key-door: the goal with the key ends the episode with +4");
    bool keysep = true;
    for (int k = 0; k < 300; k++) { wk.reset(scenario(13)); if (std::hypot(wk.kx - wk.x, wk.ky - wk.y) < .5f || std::hypot(wk.kx - wk.gx, wk.ky - wk.gy) < .5f) keysep = false; }
    CHECK(keysep, "key-door: the key never spawns near the agent or the goal");
    {  // task family: chain, invert, moving goal
      World wc(31);
      wc.reset(scenario(15));
      wc.sparse = true; wc.windx = wc.windy = 0; wc.vx = wc.vy = 0;
      wc.x = wc.gx; wc.y = wc.gy;
      Transition c0 = wc.step(0);
      CHECK(!c0.done && wc.stage == 0 && c0.ns[8] == -1.f, "chain: the goal is inert before both keys");
      wc.x = wc.k2x; wc.y = wc.k2y; wc.vx = wc.vy = 0;
      Transition c1 = wc.step(0);
      CHECK(wc.stage == 0 && c1.r < 0.f, "chain: the second key does nothing before the first");
      wc.x = wc.kx; wc.y = wc.ky; wc.vx = wc.vy = 0;
      Transition c2 = wc.step(0);
      CHECK(wc.stage == 1 && std::abs(c2.r - .99f) < 1e-3f && c2.ns[8] == 0.f, "chain: first key gives +1 and advances the stage");
      wc.x = wc.k2x; wc.y = wc.k2y; wc.vx = wc.vy = 0;
      Transition c3 = wc.step(0);
      CHECK(wc.stage == 2 && std::abs(c3.r - .99f) < 1e-3f && c3.ns[6] == 0.f && c3.ns[8] == 1.f, "chain: second key gives +1 and the key vector vanishes");
      wc.x = wc.gx; wc.y = wc.gy; wc.vx = wc.vy = 0;
      Transition c4 = wc.step(0);
      CHECK(c4.done && std::abs(c4.r - 3.99f) < 1e-3f, "chain: the goal after both keys gives +4 and ends the episode");
      bool sep = true;
      for (int k = 0; k < 300; k++) {
        wc.reset(scenario(16));
        auto d = [](float a, float b, float c, float e) { return std::hypot(a - c, b - e); };
        if (d(wc.kx, wc.ky, wc.x, wc.y) < .5f || d(wc.k2x, wc.k2y, wc.x, wc.y) < .5f || d(wc.kx, wc.ky, wc.gx, wc.gy) < .5f || d(wc.k2x, wc.k2y, wc.gx, wc.gy) < .5f || d(wc.kx, wc.ky, wc.k2x, wc.k2y) < .5f) sep = false;
      }
      CHECK(sep, "chain: keys never spawn close to the agent, the goal or each other");
      World wv(32);
      wv.reset(scenario(17));
      wv.sparse = true; wv.windx = wv.windy = 0; wv.vx = wv.vy = 0;
      wv.step(1);
      const float vx_before = wv.vx;
      wv.vx = wv.vy = 0; wv.has_key = true;
      wv.step(1);
      CHECK(vx_before > .09f && wv.vx < -.09f, "invert: thrust direction mirrors after the key");
      World wg(33);
      wg.reset(scenario(19));
      wg.windx = wg.windy = 0;
      const float g0x = wg.gx, g0y = wg.gy;
      wg.x = -.9f; wg.y = -.9f; wg.vx = wg.vy = 0;
      wg.step(0);
      bool moved = std::abs(wg.gx - g0x) + std::abs(wg.gy - g0y) > .02f, inside = true;
      for (int k = 0; k < 600; k++) { wg.x = -.9f; wg.y = -.9f; wg.step(0); if (std::abs(wg.gx) > .82f + 1e-4f || std::abs(wg.gy) > .82f + 1e-4f) inside = false; }
      CHECK(moved && inside, "moving goal: it drifts and bounces inside the arena");
    }
    {  // multi-step training: analytic gradient of the unrolled loss vs finite differences
      const bool ctx_saved = cfg.ctx;
      bool all_ok = true;
      double worst = 0;
      for (int use_ctx = 0; use_ctx < 2; use_ctx++) {
        cfg.ctx = use_ctx == 1;
        Agent ag(77);
        World wu(78);
        wu.reset(scenario(0));
        Ctx c;
        std::mt19937 ar(79);
        for (int k = 0; k < 14; k++) {
          Transition tr = wu.step((int)(ar() % ACT));
          tr.c = c;
          c = next_ctx(tr.s, tr.ns, tr.a, c);
          ag.replay.push_back(tr);
        }
        MLP m = ag.models[0];
        for (auto& w : m.w3) w *= 5.f;  // make the dynamics non-trivial so the chained terms matter
        const int K = 4;
        Grad g;
        ag.unroll_pass(m, 2, K, &g);
        auto num = [&](float& p) {
          const float keep = p, eps = 1e-2f;  // large enough to stay clear of float rounding noise (error shrinks with eps)
          p = keep + eps; double lp = ag.unroll_pass(m, 2, K, nullptr);
          p = keep - eps; double lm = ag.unroll_pass(m, 2, K, nullptr);
          p = keep;
          return (lp - lm) / (2.0 * eps);
        };
        auto check = [&](float& p, float an) {
          double nu = num(p), den = std::max(1e-3, std::max(std::abs(nu), (double)std::abs(an)));
          double err = std::abs(nu - an) / den;
          worst = std::max(worst, err);
          if (err > .03) all_ok = false;
        };
        for (int i : {0, 7, 33, 100, 160, 300}) check(m.w1[i % (H1 * IN)], g.w1[i % (H1 * IN)]);
        for (int i : {0, 5, 20, 44}) { check(m.b1[i], g.b1[i]); check(m.b2[i % H2], g.b2[i % H2]); }
        for (int i : {1, 90, 700, 2000}) check(m.w2[i % (H2 * H1)], g.w2[i % (H2 * H1)]);
        for (int i : {3, 50, 300, 500}) check(m.w3[i % (OUT * H2)], g.w3[i % (OUT * H2)]);
        // columns that read the velocity-change context and the previous-action one-hot (the chained inputs)
        for (int j : {1, 9, 30}) { check(m.w1[j * IN + OBS], g.w1[j * IN + OBS]); check(m.w1[j * IN + 2], g.w1[j * IN + 2]); }
      }
      cfg.ctx = ctx_saved;
      CHECK(all_ok, "multi-step training: analytic gradient matches finite differences (worst rel. err " + std::to_string(worst) + ")");
    }
    {  // procedural task family
      bool det = true, solvable = true, valid = true;
      int cnt[3] = {0, 0, 0}, inv = 0, mov = 0, hid = 0;
      for (int i = 0; i < 400; i++) {
        Scn a = scenario(GEN_BASE + i), b = scenario(GEN_BASE + i);
        if (a.keydoor != b.keydoor || a.chain != b.chain || a.invert != b.invert || a.gmove != b.gmove || a.sense != b.sense || a.d.sp != b.d.sp || a.d.wx != b.d.wx || std::string(a.name) != std::string(b.name)) det = false;
        if (std::abs(a.d.wx) > a.d.sp / 1.3f + 1e-6f || std::abs(a.d.wy) > a.d.sp / 1.3f + 1e-6f) solvable = false;
        if ((a.invert && !a.keydoor) || (a.keydoor && a.chain) || (a.sense > 0.f && (a.keydoor || a.chain || a.gmove || a.invert))) valid = false;
        cnt[a.chain ? 2 : a.keydoor ? 1 : 0]++; inv += a.invert; mov += a.gmove; hid += a.sense > 0.f;
        World w(900 + i);
        Vec o = w.reset(a);
        for (float v : o) if (!std::isfinite(v)) valid = false;
      }
      CHECK(det, "generator: the same index always gives the same task");
      CHECK(solvable, "generator: wind is always weaker than thrust / 1.3 (every task is solvable)");
      CHECK(valid, "generator: feature combinations are valid and observations finite");
      CHECK(cnt[0] > 60 && cnt[1] > 100 && cnt[2] > 100 && inv > 20 && mov > 50 && hid > 10, "generator: all structure types occur (plain " + std::to_string(cnt[0]) + ", key " + std::to_string(cnt[1]) + ", chain " + std::to_string(cnt[2]) + ", invert " + std::to_string(inv) + ", moving " + std::to_string(mov) + ", hidden " + std::to_string(hid) + " of 400)");
    }
    // wind flips exactly at the scenario's period
    World wf(3);
    wf.reset(scenario(5));
    float w0 = wf.windx;
    for (int k = 0; k < 49; k++) wf.step(0);
    bool before = wf.windx == w0;
    wf.step(0);
    CHECK(before && wf.windx == -w0, "wind flips at tick 50 in scenario N2-gusty");
  }

  std::cout << "[4] learning components\n";
  {
    // Q-learning on a 6-state chain: action 1 moves right, reward 4 on entering state 5 (terminal).
    cfg = Cfg{};
    cfg.gamma = .9f;
    cfg.value = true;
    Agent ag(31);
    for (int rep = 0; rep < 60; rep++)
      for (int k = 0; k < 5; k++)
        for (int a = 0; a < 2; a++) {
          Transition t;
          t.s[0] = k / 10.f;
          t.a = a;
          bool right = a == 1;
          t.ns[0] = (right ? k + 1 : k) / 10.f;
          t.done = right && k + 1 == 5;
          t.r = t.done ? 4.f : 0.f;
          ag.remember(t);
          ag.steps++;
        }
    for (int i = 0; i < 9000; i++) ag.learn_q();
    bool ok = true;
    std::string det;
    for (int k = 0; k < 5; k++) {
      Vec s{};
      s[0] = k / 10.f;
      auto q = ag.qn.forward(make_in(s, Ctx{}));
      float want1 = 4.f * std::pow(.9f, 4 - k), want0 = .9f * want1;
      float got1 = q[1] * QSCALE, got0 = q[0] * QSCALE;
      det += " k" + std::to_string(k) + ":" + std::to_string(got1).substr(0, 4) + "/" + std::to_string(want1).substr(0, 4);
      if (std::abs(got1 - want1) > .12f * want1 + .05f || std::abs(got0 - want0) > .12f * want0 + .05f) ok = false;
    }
    CHECK(ok, "double-Q converges to gamma^k * 4 on the chain (Q(right) got/want:" + det + ")");

    // RND: states seen in training become less novel than far-away states.
    cfg = Cfg{};
    cfg.rnd_beta = .05f;
    Agent rg(32);
    for (int i = 0; i < 400; i++) {
      Transition t;
      for (auto& v : t.ns) v = .1f * U(R);
      rg.remember(t);
    }
    for (int i = 0; i < 1500; i++) rg.learn_rnd();
    Vec in{}, far{};
    for (auto& v : in) v = .05f;
    for (auto& v : far) v = .9f;
    float ni = rg.novelty(in), nf = rg.novelty(far);
    CHECK(nf > 3.f * ni && ni < 1.5f, "RND: novelty far=" + std::to_string(nf) + " vs seen=" + std::to_string(ni));

    // pos_frac oversampling really draws goal transitions
    cfg = Cfg{};
    cfg.pos_frac = .5f;
    Agent pg(33);
    for (int i = 0; i < 1000; i++) { Transition t; t.done = (i % 100 == 0); pg.remember(t); }
    int hit = 0;
    for (int i = 0; i < 4000; i++) hit += pg.replay[pg.sample_idx()].done ? 1 : 0;
    CHECK(hit > 1700 && hit < 2300, "pos_frac=0.5 samples ~50% goal transitions (got " + std::to_string(hit * 100 / 4000) + "%)");
    CHECK(pg.pos_idx.size() == 10, "positive index list holds 10 goal transitions");
  }

  std::cout << "[4b] echo-state memory\n";
  {
    cfg = Cfg{};
    cfg.reservoir = true;
    Ctx c0;
    Vec o{}, zero{};
    Ctx ca = c0, cb = c0;
    Vec oa = o, ob = o;
    oa[4] = .5f; oa[5] = -.3f; ob[4] = -.5f; ob[5] = .3f;     // two different "goal seen" pulses
    ca = next_ctx(zero, oa, 0, ca); cb = next_ctx(zero, ob, 0, cb);
    auto dist = [](const Ctx& a, const Ctx& b) { double d = 0; for (int i = 0; i < MEM; i++) d += sq(a.m[i] - b.m[i]); return std::sqrt(d); };
    double d0 = dist(ca, cb), maxabs = 0;
    for (int k = 0; k < 40; k++) { ca = next_ctx(zero, zero, 1, ca); cb = next_ctx(zero, zero, 1, cb); for (float v : ca.m) maxabs = std::max(maxabs, (double)std::abs(v)); }
    double d40 = dist(ca, cb);
    CHECK(d0 > 1e-4 && d40 > .05 * d0, "memory still distinguishes the two pulses 40 steps later (" + std::to_string(d40 / d0 * 100) + "% of the initial gap)");
    CHECK(maxabs <= 1.0, "memory state stays bounded in [-1,1]");
    Ctx cc = c0;
    cc = next_ctx(zero, oa, 0, cc);
    Ctx cd = c0;
    cd = next_ctx(zero, oa, 0, cd);
    CHECK(cc.m == cd.m, "memory update is deterministic");
    cfg.reservoir = false;
    Ctx ce = next_ctx(zero, oa, 0, c0);
    CHECK(ce.m == c0.m, "memory stays zero when the reservoir is off");
  }

  std::cout << "[5] checkpoint save/load\n";
  {
    cfg = Cfg{};
    cfg.value = true;
    cfg.rnd_beta = .05f;
    cfg.reservoir = true;
    Agent A(41);
    World w(42);
    Scn sc = scenario(0);
    Vec s = w.reset(sc);
    Ctx c;
    for (int i = 0; i < 900; i++) {
      Transition tr = w.step((int)(R() % ACT));
      tr.c = c;
      A.learn(tr);
      c = next_ctx(tr.s, tr.ns, tr.a, c);
      s = tr.ns;
      if (tr.done || i % 150 == 149) { s = w.reset(sc); c = Ctx{}; }
    }
    A.episodes_done = 7; A.goals = 3;
    const std::string path = "nx_selftest.ckpt";
    CHECK(A.save(path), "save succeeds");
    Agent B(99);
    CHECK(B.load(path), "load succeeds");
    bool same = A.replay.size() == B.replay.size() && A.steps == B.steps && A.updates == B.updates &&
                A.episodes_done == B.episodes_done && A.goals == B.goals && A.pos_idx == B.pos_idx &&
                A.rnd_ema == B.rnd_ema && A.rnd_ready == B.rnd_ready;
    for (size_t i = 0; i < A.replay.size(); i += 37)
      if (A.replay[i].s != B.replay[i].s || A.replay[i].ns != B.replay[i].ns || A.replay[i].a != B.replay[i].a ||
          A.replay[i].r != B.replay[i].r || A.replay[i].done != B.replay[i].done || A.replay[i].c.pa != B.replay[i].c.pa ||
          A.replay[i].c.dv[0] != B.replay[i].c.dv[0] || A.replay[i].c.m != B.replay[i].c.m) same = false;
    for (int k = 0; k < 20; k++) {
      Vec v{};
      for (auto& e : v) e = U(R);
      Ctx cc;
      cc.dv[0] = U(R); cc.dv[1] = U(R); cc.pa = k % ACT;
      for (int m = 0; m < ENSEMBLE; m++) if (A.models[m].forward(make_in(v, cc), k % ACT, cc.pa) != B.models[m].forward(make_in(v, cc), k % ACT, cc.pa)) same = false;
      if (A.qn.forward(make_in(v, cc)) != B.qn.forward(make_in(v, cc))) same = false;
      if (A.qt.forward(make_in(v, cc)) != B.qt.forward(make_in(v, cc))) same = false;
      if (A.rnd_p.forward(v) != B.rnd_p.forward(v) || A.rnd_t.forward(v) != B.rnd_t.forward(v)) same = false;
    }
    CHECK(same, "loaded agent is bit-identical (models, Q, RND, replay, counters)");
    // truncated and corrupted files must be rejected without touching the agent
    std::ifstream in(path, std::ios::binary);
    std::string bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    in.close();
    { std::ofstream o("nx_selftest_trunc.ckpt", std::ios::binary); o.write(bytes.data(), (std::streamsize)(bytes.size() / 2)); }
    { std::string bad = bytes; bad[2] = 'Z'; std::ofstream o("nx_selftest_magic.ckpt", std::ios::binary); o.write(bad.data(), (std::streamsize)bad.size()); }
    Agent C(5);
    size_t before = C.replay.size();
    uint64_t st0 = C.steps;
    bool r1 = C.load("nx_selftest_trunc.ckpt"), r2 = C.load("nx_selftest_magic.ckpt"), r3 = C.load("does_not_exist.ckpt");
    CHECK(!r1 && !r2 && !r3 && C.replay.size() == before && C.steps == st0, "truncated / corrupted / missing checkpoints are rejected, agent untouched");
    std::error_code ec;
    std::filesystem::remove(path, ec);
    std::filesystem::remove("nx_selftest_trunc.ckpt", ec);
    std::filesystem::remove("nx_selftest_magic.ckpt", ec);
  }

  std::cout << "[6] planner\n";
  {
    // With the TRUE reward and a perfectly informative start, the planner must move toward the goal.
    cfg = Cfg{};
    cfg.oracle = false;
    Agent ag(51);
    World w(52);
    Scn sc = scenario(0);
    // train the model briefly on random-walk data (enough for a one-step sanity check, not for control)
    Vec s = w.reset(sc);
    Ctx c;
    // NOTE: episodes must be full length (MAX_STEPS) in both training and testing, otherwise the wind
    // flip at tick 100 is never seen in training and the held-out error is meaningless.
    for (int i = 0; i < 6000; i++) {
      Transition tr = w.step((int)(R() % ACT));
      tr.c = c;
      ag.learn(tr);
      c = next_ctx(tr.s, tr.ns, tr.a, c);
      if (tr.done || i % MAX_STEPS == MAX_STEPS - 1) { w.reset(sc); c = Ctx{}; }
    }
    double merr = 0, perr = 0;
    int n = 0;
    Vec s2 = w.reset(sc);
    Ctx c2;
    for (int i = 0; i < 4 * MAX_STEPS; i++) {
      Transition tr = w.step((int)(R() % ACT));
      tr.c = c2;
      merr += ag.state_error(tr);
      double pe = 0;
      for (int j = 0; j < OBS; j++) pe += sq(tr.ns[j] - tr.s[j]);
      perr += pe / OBS;
      n++;
      c2 = next_ctx(tr.s, tr.ns, tr.a, c2);
      if (tr.done || i % MAX_STEPS == MAX_STEPS - 1) { w.reset(sc); c2 = Ctx{}; }
    }
    CHECK(merr < .5 * perr, "world model beats the 'nothing changes' baseline on fresh data (" + std::to_string(merr / n) + " vs " + std::to_string(perr / n) + ")");
    (void)s2; (void)s;
    cfg.horizon = 12; cfg.pop = 40; cfg.iters = 3;
    Vec obs = w.reset(sc);
    int act = ag.choose(obs, Ctx{}, 1000, false);
    CHECK(act >= 0 && act < ACT, "planner returns a valid action with horizon 12 / pop 40 / 3 iterations");
  }

  {  // stabilisers: weight blending, learning-rate schedule, averaged-weight swap
    MLP a(11), b(12);
    const float a0 = a.w1[3], b0 = b.w1[3];
    MLP c = a; c.blend(b, 0.f);
    MLP d = a; d.blend(b, 1.f);
    MLP e = a; e.blend(b, .5f);
    CHECK(c.w1[3] == a0 && d.w1[3] == b0 && std::abs(e.w1[3] - .5f * (a0 + b0)) < 1e-6f, "MLP::blend: tau 0 keeps, tau 1 copies, tau .5 is the midpoint");
    Agent ag(5);
    cfg.lr_from = 10; cfg.lr_to = 20;
    ag.episodes_done = 0;  const float m0 = ag.lr_mult();
    ag.episodes_done = 15; const float m1 = ag.lr_mult();
    ag.episodes_done = 40; const float m2 = ag.lr_mult();
    cfg.lr_from = cfg.lr_to = 0;
    CHECK(m0 == 1.f && std::abs(m1 - .55f) < 1e-6f && std::abs(m2 - .1f) < 1e-6f && ag.lr_mult() == 1.f, "lr schedule: 1x before A, linear to 0.1x at B, off when unset");
    ag.avg_models = ag.models; ag.avg_q = ag.qn; ag.avg_ready = true;
    const float w0 = ag.models[0].w1[0];
    ag.avg_models[0].w1[0] = w0 + 1.f;
    ag.avg_swap();
    const bool in1 = ag.avg_in && ag.models[0].w1[0] == w0 + 1.f;
    ag.avg_swap();
    CHECK(in1 && !ag.avg_in && ag.models[0].w1[0] == w0 && ag.avg_models[0].w1[0] == w0 + 1.f, "averaged weights swap in and out without losing either set");
  }

  {  // v9 item 1: symbols and next-token model
    std::mt19937 rg(77);
    auto mk = [&](int n, int K, int amax) {  // transitions on one grid dimension (dim 0) and one slow dimension (dim 2); everything else constant
      std::vector<Transition> rp;
      for (int i = 0; i < n; i++) {
        Transition z;
        const int k = (int)(rg() % (unsigned)K), a = (int)(rg() % (unsigned)amax), k2 = std::min(K - 1, k + 1 + a);
        z.s[0] = -1.f + 2.f * (float)k / (float)K; z.ns[0] = -1.f + 2.f * (float)k2 / (float)K;
        z.s[2] = z.ns[2] = (k % 2) ? .5f : -.5f;
        z.a = a; z.r = (k2 == K - 1) ? 1.f : -.01f;
        rp.push_back(z);
      }
      return rp;
    };
    {
      TokenModel T; T.D = 4; T.H = 8; T.RH = 6;
      auto rp = mk(60, 8, 2);
      T.init(rp, 8, 5);
      bool exact = T.na() == 2 && T.ad[0] == 0 && T.ad[1] == 2;
      for (int k = 0; k < 8; k++) exact = exact && T.tok(0, -1.f + 2.f * (float)k / 8.f) == k;
      std::mt19937 r2(9);
      for (int k = 0; k < 500; k++) { Vec v{}; v[0] = (float)((int)(r2() % 2000) - 1000) / 1000.f; v[2] = (float)((int)(r2() % 2000) - 1000) / 1000.f; T.observe(v); }
      bool sorted = true;
      for (int j : T.ad) for (int k = 1; k < 8; k++) sorted = sorted && T.cb[j][k] > T.cb[j][k - 1];
      CHECK(exact, "VQ: only varying dimensions are active, grid values map to their own token");
      CHECK(sorted, "VQ: centroids stay strictly ordered after online k-means updates");
      // gradient check on a few parameters of every part
      const Transition z = rp[3];
      T.accumulate(z, true);
      struct P { float* p; float g; const char* nm; };
      std::vector<P> ps = {{&T.ld.w[5], T.ld.gw[5], "ld.w"}, {&T.ld.b[3], T.ld.gb[3], "ld.b"}, {&T.l2.w[7], T.l2.gw[7], "l2.w"}, {&T.l1.w[11], T.l1.gw[11], "l1.w"},
                           {&T.r1.w[4], T.r1.gw[4], "r1.w"}, {&T.r2.w[2], T.r2.gw[2], "r2.w"}};
      int zi[OBS]; T.toks(z.s, zi);
      ps.push_back({&T.emb[0][(size_t)zi[0] * T.D + 1], T.gemb[0][(size_t)zi[0] * T.D + 1], "emb"});
      int bad = 0; double worst = 0;
      for (auto& q : ps) {
        const float o = *q.p, e = 1e-2f;
        *q.p = o + e; const double lp = T.accumulate(z, false);
        *q.p = o - e; const double lm = T.accumulate(z, false);
        *q.p = o;
        const double num = (lp - lm) / (2.0 * e), err = std::abs(num - q.g) / std::max(1e-3, std::abs(num) + std::abs((double)q.g));
        worst = std::max(worst, err);
        if (err > 2e-2 && std::abs(num - q.g) > 2e-4) bad++;
      }
      CHECK(bad == 0, "token model: gradient check on all layers and the embeddings (worst relative diff " + std::to_string(worst) + ")");
    }
    {  // a dimension that is an exact linear function of two others is found and derived, not predicted
      TokenModel T; T.D = 4; T.H = 8; T.RH = 6;
      std::vector<Transition> rp;
      for (int i = 0; i < 400; i++) {
        Transition z; const int kv = (int)(rg() % 64u), kt = (int)(rg() % 64u), kv2 = std::min(63, kv + 3);
        auto g = [](int k) { return -1.f + (float)k / 32.f; };
        z.s[0] = g(kv); z.ns[0] = g(kv2); z.s[2] = z.ns[2] = g(kt);
        z.s[4] = clampf(z.s[2] - z.s[0], -1.f, 1.f); z.ns[4] = clampf(z.ns[2] - z.ns[0], -1.f, 1.f);
        z.s[1] = (float)(i % 7) / 7.f; z.ns[1] = (float)((i + 1) % 7) / 7.f;
        rp.push_back(z);
      }
      T.init(rp, 64, 3);
      CHECK(T.ready && T.der[4] && !T.der[0] && !T.der[2] && !T.der[1] && T.pd.size() == 3, "token model: finds 'target - current' as a derived dimension and predicts only the others");
    }
    {  // it really learns a small deterministic arithmetic (v' = v + 1 + a, two actions) and the reward at the end
      TokenModel T; T.D = 8; T.H = 48; T.RH = 16;
      const int K = 16;
      auto rp = mk(400, K, 2);
      T.init(rp, K, 6);
      for (int it = 0; it < 1500; it++) {
        for (int b = 0; b < 32; b++) T.accumulate(rp[rg() % rp.size()], true);
        T.step_all(3e-3f, 1.f / 32.f);
      }
      int ok = 0, tot = 0, rok = 0;
      for (int k = 0; k < K - 1; k++) for (int a = 0; a < 2; a++) {
        Vec v{}; v[0] = -1.f + 2.f * (float)k / (float)K; v[2] = (k % 2) ? .5f : -.5f;
        Vec ns; float r; T.predict(v, a, ns, r);
        const int k2 = std::min(K - 1, k + 1 + a);
        tot++; ok += T.tok(0, ns[0]) == k2; rok += (r > .5f) == (k2 == K - 1);
      }
      CHECK(ok >= tot - 1 && rok >= tot - 1, "token model learns v' = v + 1 + a and the terminal reward (" + std::to_string(ok) + "/" + std::to_string(tot) + " next tokens, " + std::to_string(rok) + "/" + std::to_string(tot) + " rewards)");
      std::stringstream ss; T.save(ss);
      TokenModel U; const bool loaded = U.load(ss);
      bool same = loaded && U.ready && U.ad == T.ad;
      for (int k = 0; k < K && same; k++) { Vec v{}; v[0] = -1.f + 2.f * (float)k / (float)K; v[2] = .5f; Vec n1, n2; float r1v, r2v; T.predict(v, 1, n1, r1v); U.predict(v, 1, n2, r2v); same = same && n1 == n2 && r1v == r2v; }
      CHECK(same, "token model: save/load reproduces predictions exactly");
    }
  }

  {  // v9 item 2: concepts and manager
    std::mt19937 rg(5);
    std::vector<Transition> rp;
    const float cx[3] = {-.6f, 0.f, .6f}, cy[3] = {-.5f, .5f, -.2f};
    for (int i = 0; i < 900; i++) { Transition z; const int k = i % 3; z.ns[0] = cx[k] + .03f * (float)((int)(rg() % 100) - 50) / 50.f; z.ns[1] = cy[k] + .03f * (float)((int)(rg() % 100) - 50) / 50.f; z.s = z.ns; rp.push_back(z); }
    Concepts C; C.init(rp, 3, 11);
    bool found = C.ready && C.dims.size() == 2;
    for (int k = 0; k < 3; k++) { Vec v{}; v[0] = cx[k]; v[1] = cy[k]; float d; C.nearest(v, &d); found = found && d < .2f; }
    Vec va{}, vb{}; va[0] = cx[0]; va[1] = cy[0]; vb[0] = cx[1]; vb[1] = cy[1];
    CHECK(found && C.nearest(va) != C.nearest(vb), "concepts: k-means finds the three clusters, each state maps to its own concept");
    std::stringstream ss; C.save(ss); Concepts C2; const bool okl = C2.load(ss);
    CHECK(okl && C2.NC == 3 && C2.nearest(va) == C.nearest(va) && C2.dist(va, vb) == C.dist(va, vb), "concepts: save/load roundtrip");
    QHi Q; Q.init(3, 4); Vec s0{}; s0[0] = .3f; s0[1] = -.2f; float q[3];
    Q.forward(s0, q);
    float dq[3] = {0, q[1] - .7f, 0};
    Q.backward(s0, dq);
    const float ga = Q.b.gw[9], gb = Q.c.gb[1];
    auto loss = [&]() { float qq[3]; Q.forward(s0, qq); return .5 * (double)(qq[1] - .7f) * (qq[1] - .7f); };
    const float o1 = Q.b.w[9], e = 1e-2f; Q.b.w[9] = o1 + e; const double lp = loss(); Q.b.w[9] = o1 - e; const double lm = loss(); Q.b.w[9] = o1;
    const double num = (lp - lm) / (2 * e);
    CHECK(std::abs(num - ga) < 2e-3 + 2e-2 * std::abs(num) && std::abs(gb - (q[1] - .7f)) < 1e-6, "manager value net: gradient check");
    // the manager learns which concept pays: option 2 gives +1, the others 0
    Agent A(3); cfg.hier = 3; A.cn = C; A.qh.init(4, 31u); A.qht = A.qh;
    for (int i = 0; i < 300; i++) { OptTr o; o.s0 = va; o.s1 = vb; o.g = (int)(rg() % 3u); o.R = o.g == 2 ? 1.f : 0.f; o.disc = .9f; o.done = true; A.ob.push_back(o); }
    Transition tt; tt.s = va; tt.ns = va; A.replay.assign(700, tt);
    for (int i = 0; i < 600; i++) A.learn_hier(tt);
    float qq[4]; A.qh.forward(va, qq);
    CHECK(qq[2] > qq[0] + .5f && qq[2] > qq[1] + .5f, "manager: option values converge to the true option returns (" + std::to_string(qq[0]) + ", " + std::to_string(qq[1]) + ", " + std::to_string(qq[2]) + ")");
  }

  {  // v9 item 3: the error map learns where the model is wrong
    std::mt19937 rg(21);
    Agent A(5);
    std::vector<Transition> rp;
    for (int i = 0; i < 400; i++) {
      Transition z; z.s[0] = (i % 2) ? .5f : -.5f; z.a = i % ACT; z.ns = z.s;
      if (i % 2) z.ns[0] += .5f;  // large surprise in the right half only
      rp.push_back(z);
    }
    A.replay = rp;
    for (int i = 0; i < 700; i++) A.learn_err();
    Vec l{}, r{}; l[0] = -.5f; r[0] = .5f;
    const float el = A.en.forward(make_in(l, Ctx{}))[0], er = A.en.forward(make_in(r, Ctx{}))[1];
    CHECK(er > el + .5f, "curiosity: the error map is high where the model is wrong and low where it is right (" + std::to_string(el) + " vs " + std::to_string(er) + ")");
  }
  {  // v9 item 4: reasoning on the graph, pinned exemplars, persistence
    Graph G; G.NC = 5; G.ready = true; G.cc.NC = 5; G.cc.ready = true; G.cc.dims = {0}; G.cc.sd.fill(1.f);
    for (int k = 0; k < 5; k++) { Vec v{}; v[0] = (float)k * .2f; G.cc.c.push_back(v); }
    G.cc.cnt.assign(5, 1.f); G.cc.rad.assign(5, .1f);
    G.adj.assign(5 * ACT, {}); G.rm.assign(5 * ACT, 0.f); G.rc = G.rm; G.tm_ = G.rm; G.V.assign(5, 0.f);
    // node 0 --a0--> node 1 --a0--> node 4 (goal, +1, terminal);  node 0 --a1--> node 2 --a0--> node 3 --a0--> node 4
    for (int i = 0; i < 60; i++) {
      G.observe_nodes(0, 1, 0, -.01f, false); G.observe_nodes(1, 4, 0, 1.f, true);
      G.observe_nodes(0, 2, 1, -.01f, false); G.observe_nodes(2, 3, 0, -.01f, false); G.observe_nodes(3, 4, 0, 1.f, true);
    }
    G.solve(.94f);
    const int before = G.best_action(0, .94f);
    const float v0 = G.V[0];
    for (int i = 0; i < 700; i++) G.observe_nodes(0, 0, 0, -.01f, false);  // the short road is now blocked: the agent stays where it is
    G.solve(.94f);
    const int after = G.best_action(0, .94f);
    CHECK(before == 0 && after == 1 && G.V[0] > .5f * v0, "graph reasoning: the short road is preferred, and after it is blocked the value switches to the detour");
    std::stringstream ss; G.save(ss); Graph H; const bool okl = H.load(ss);
    CHECK(okl && H.V == G.V && H.best_action(0, .94f) == after && H.adj[0].size() == G.adj[0].size(), "graph: save/load roundtrip");
    // pinned exemplars survive a flood of newer transitions
    Cfg keep = cfg; cfg.graph = 5; cfg.rcap = 1000;
    Agent P(2); P.gr = G;
    auto mkt = [&](int from, int to, int a, float r) { Transition t; t.s[0] = (float)from * .2f; t.ns[0] = (float)to * .2f; t.a = a; t.r = r; return t; };
    P.remember(mkt(1, 4, 0, 1.f)); P.remember(mkt(2, 3, 0, 0.f));
    for (int i = 0; i < 4000; i++) P.remember(mkt(0, 0, 3, 0.f));  // flood of one single edge
    int survived = 0;
    for (const auto& z : P.replay) if (z.ns[0] > .5f && z.a == 0) survived++;
    CHECK(survived >= 2 && P.replay.size() == 1000 && P.pin_count >= 3, "graph memory: pinned exemplars of rare edges survive a flood of newer transitions (" + std::to_string(survived) + " kept, " + std::to_string(P.pin_count) + " pinned)");
    cfg = keep;
  }
  {  // all v9 components through a checkpoint
    Cfg keep = cfg; cfg.vq = 8; cfg.hier = 4; cfg.curious = 1.f; cfg.graph = 4; cfg.rcap = 1000; cfg.value = true; cfg.heur = false;
    std::mt19937 rg(33);
    Agent A(7);
    World w(3);
    for (int i = 0; i < 700; i++) {
      Transition t; t.s[0] = (float)((int)(rg() % 8u)) / 4.f - 1.f; t.s[1] = (float)((int)(rg() % 4u)) / 2.f - 1.f; t.a = (int)(rg() % 3u);
      t.ns = t.s; t.ns[0] = clampf(t.s[0] + .25f, -1.f, 1.f); t.r = t.ns[0] >= .99f ? 1.f : -.01f; t.done = t.r > .5f;
      A.learn(t);
    }
    const std::string path = "nx_selftest_v9.ckpt";
    const bool sv = A.save(path);
    Agent B(7);
    const bool ld = B.load(path);
    std::remove(path.c_str());
    Vec v{}; v[0] = -.25f; v[1] = .5f; Vec n1, n2; float r1, r2, u1, u2;
    bool same = sv && ld && A.tm.ready && B.tm.ready && A.cn.ready && B.cn.ready && A.gr.ready && B.gr.ready;
    if (same) { A.predict(v, Ctx{}, 1, n1, r1, u1); B.predict(v, Ctx{}, 1, n2, r2, u2); same = n1 == n2 && r1 == r2 && A.gr.V == B.gr.V && A.cn.c.size() == B.cn.c.size() && A.pinmap == B.pinmap && A.pin == B.pin && A.ring == B.ring && A.en.forward(make_in(v, Ctx{})) == B.en.forward(make_in(v, Ctx{})); }
    CHECK(same, "checkpoint with symbols, manager, error map and graph memory reloads identically");
    cfg = keep;
  }

  cfg = saved;
  // ---- v10: rule synthesis ----
  {
    Synth S;
    std::vector<Transition> rep;
    std::mt19937 G(5);
    auto lat = [](int n) { return (float)n / 128.f - 1.f; };
    for (int k = 0; k < 4000; k++) {
      Transition t{}; const int mv = (int)(G() % 256), a = (int)(G() % 9);
      int nv = mv;
      switch (a) { case 0: nv = mv + 1; break; case 1: nv = mv + 2; break; case 2: nv = mv + 4; break; case 3: nv = mv + 8; break;
                   case 4: nv = mv * 2; break; case 5: nv = mv * 3; break; case 6: nv = mv - 1; break; case 7: nv = mv - 4; break; default: nv = (mv % 2 == 0) ? mv / 2 : mv; }
      if (!(nv >= 0 && nv <= 255)) nv = mv;
      t.a = a; t.s[0] = lat(mv); t.ns[0] = lat(nv); rep.push_back(t);
    }
    S.hyp.push_back({4.f, 0.f, 2.f, 1.f});    // true: x2 is  2n+1  in the normalised unit
    S.hyp.push_back({0.f, 0.f, 1.f, 0.5f});   // false
    S.fit(rep);
    bool allok = true; int bad = 0;
    for (int k = 0; k < 2000; k++) {
      const int mv = (int)(G() % 256), a = (int)(G() % 9);
      int nv = mv;
      switch (a) { case 0: nv = mv + 1; break; case 1: nv = mv + 2; break; case 2: nv = mv + 4; break; case 3: nv = mv + 8; break;
                   case 4: nv = mv * 2; break; case 5: nv = mv * 3; break; case 6: nv = mv - 1; break; case 7: nv = mv - 4; break; default: nv = (mv % 2 == 0) ? mv / 2 : mv; }
      if (!(nv >= 0 && nv <= 255)) nv = mv;
      if (!S.has(a, 0)) { allok = false; continue; }
      if (std::fabs(S.next(a, 0, lat(mv)) - lat(nv)) > 2e-3f) bad++;
    }
    CHECK(allok && bad == 0, "synth: all nine NumberChain operators recovered exactly (incl. parity guard of /2 and range guards), 0 errors in 2000 fresh states");
    CHECK(S.hyp_ok == 1 && S.hyp_bad == 1, "synth: a true rule hypothesis is confirmed by experience, a false one is refuted");
    Synth E; std::vector<Transition> few(rep.begin(), rep.begin() + 40); E.fit(few);
    CHECK(E.nrules == 0, "synth: too few samples per action -> rules withheld");
  }
  {  // v10: bottleneck sub-goals = concepts that successful episodes pass through clearly more often than failed ones
    Cfg keep = cfg; cfg.bottleneck = true;
    Agent A(9); A.cn.NC = 5; A.cn.ready = true;
    for (int e = 0; e < 40; e++) {
      A.ep_seen.assign(5, 0); A.ep_seen[0] = 1;
      const bool succ = e % 2 == 0;
      if (succ) A.ep_seen[2] = 1; else { A.ep_seen[1] = 1; A.ep_seen[3] = 1; }
      A.ep_succ = succ;
      A.bn_finish_episode();
    }
    CHECK(A.bn_ok[2] == 1 && A.bn_ok[0] == 0 && A.bn_ok[1] == 0 && A.bn_ok[3] == 0, "bottleneck: only the concept unique to successful episodes becomes a sub-goal candidate");
    cfg = keep;
  }
  {  // v11: the new games behave as specified
    World w(5);
    w.reset(scenario(GAME_BASE + 3)); w.sw = 0b00000000;
    Transition a = w.step(0);  // press 0 flips lights 0 and 1
    CHECK(w.sw == 0b00000011 && !a.done, "Switches: a press flips the button and its neighbours");
    w.sw = 0b11111110; Transition b = w.step(0);
    CHECK(w.sw == 0b11111101 && !b.done, "Switches: edge button flips two lights");
    w.sw = 0b11111100; Transition c = w.step(0);
    CHECK(c.done && w.sw == 255 && std::fabs(c.r - 3.99f) < 1e-3f, "Switches: all lights on ends the episode with +4");
    w.reset(scenario(GAME_BASE + 4)); w.mv = 96; w.step(0);
    CHECK(w.mv == 0, "ModChain: 96 + 1 wraps to 0 modulo 97");
    w.mv = 40; w.mt = 80; Transition d = w.step(3);
    CHECK(w.mv == 80 && d.done, "ModChain: 40 x 2 = 80 reaches the target");
    int ok = 0;
    for (int ep = 0; ep < 20; ep++) {
      World p(100 + ep); p.reset(scenario(GAME_BASE + 2));
      for (int t = 0; t < GAME_STEPS; t++) { Transition tr = p.step(greedy_action(p)); if (tr.done) { ok++; break; } }
    }
    CHECK(ok >= 18, "Pendulum: the reference controller swings up in " + std::to_string(ok) + "/20 episodes");
    World q(9); q.reset(scenario(GAME_BASE + 2)); q.pth = 0.f; q.pw = 0.f; Transition e = q.step(0);
    CHECK(std::fabs(q.pth) < 1e-6f && std::fabs(q.pw) < 1e-6f && e.done, "Pendulum: upright and still stays put (and counts as solved)");
  }
  {  // v11: formula hypotheses (closed loop with a proposer) are confirmed or refuted by experience
    Synth S;
    std::vector<Transition> rep; std::mt19937 G(8);
    std::uniform_real_distribution<float> U(-1.f, 1.f);
    for (int k = 0; k < 600; k++) { Transition t{}; t.a = 0; t.s[1] = U(G); t.s[2] = U(G); t.s[0] = U(G); t.ns[0] = clampf(.5f * t.s[1] + .1f * std::sin(2.f * t.s[2]) + .2f, -1.f, 1.f); t.ns[1] = t.s[1]; t.ns[2] = t.s[2]; rep.push_back(t); }
    std::string path = "/tmp/nexus_selftest_rules.txt";
    { std::ofstream f(path); f << "f 0 0 = 0.2 + 0.5*s1 + 0.1*sin(2*s2)\nf 0 0 = 0.2 + 0.4*s1 + 0.1*sin(2*s2)\nf 0 0 = nonsense(\n"; }
    S.load_rules(path); S.fit(rep);
    bool conf = false, ref = false, bad = false;
    for (const auto& v : S.verdicts) { if (v.rfind("CONFIRMED", 0) == 0) conf = true; if (v.rfind("REFUTED", 0) == 0) ref = true; if (v.rfind("could not parse", 0) == 0) bad = true; }
    CHECK(conf && ref && bad && S.has(0, 0), "rules file: a correct formula is confirmed and used, a wrong one refuted, a malformed line reported");
    std::remove(path.c_str());
  }
  {  // v11: reset clause ("| when <formula> < c -> v") in a formula hypothesis
    Synth S; std::vector<Transition> rep; std::mt19937 G(4);
    std::uniform_real_distribution<float> U(-1.f, 1.f);
    for (int k = 0; k < 600; k++) { Transition t{}; t.a = 0; t.s[0] = U(G); t.s[1] = U(G); t.ns[0] = t.s[0]; t.ns[1] = t.s[1] < -.5f ? 0.f : clampf(t.s[1] + .1f * t.s[0], -1.f, 1.f); rep.push_back(t); }
    std::string path = "/tmp/nexus_selftest_rules2.txt";
    { std::ofstream f(path); f << "f 0 1 = s1 + 0.1*s0 | when s1 < -0.5 -> 0\nf 0 1 = s1 + 0.1*s0\n"; }
    S.load_rules(path); S.fit(rep);
    int conf = 0, ref = 0;
    for (const auto& v : S.verdicts) { if (v.rfind("CONFIRMED", 0) == 0) conf++; if (v.rfind("REFUTED", 0) == 0) ref++; }
    CHECK(conf == 1 && ref == 1, "reset clause: the formula with the reset is confirmed, the same formula without it is refuted");
    std::remove(path.c_str());
  }
  {  // v12: rotation rules for (cos, sin) observations
    Synth S; std::vector<Transition> rep; std::mt19937 G(6);
    std::uniform_real_distribution<float> U(-1.f, 1.f), A(-3.14f, 3.14f);
    for (int k = 0; k < 800; k++) {
      Transition t{}; t.a = 0; const float ph = A(G), w = U(G);
      t.s[0] = std::cos(ph); t.s[1] = std::sin(ph); t.s[2] = w;
      const float d = .3f * w + .05f;
      t.ns[0] = std::cos(ph + d); t.ns[1] = std::sin(ph + d); t.ns[2] = w; rep.push_back(t);
    }
    Cfg keep = cfg; cfg.synth_lib = true;
    S.fit(rep);
    cfg = keep;
    bool okr = S.has(0, 0) && S.has(0, 1) && S.r[0][0].rot == 1 && S.r[0][1].rot == 2;
    double err = 0; std::mt19937 H(2);
    for (int k = 0; k < 500; k++) { const float ph = A(H), w = U(H); Vec s{}; s[0] = std::cos(ph); s[1] = std::sin(ph); s[2] = w; const float d = .3f * w + .05f;
      if (okr) err = std::max(err, (double)std::fabs(S.next(0, 0, s) - std::cos(ph + d))); }
    CHECK(okr && err < 3e-3, "rotation rule: angle increment 0.3*w + 0.05 recovered for a (cos, sin) pair, max error " + std::to_string(err));
  }
  {  // held-out family: every puzzle is solvable by the BFS oracle within the step limit, deterministic, and its observation stays in [-1, 1]
    bool solv = true, inrange = true, det = true, shown = true;
    for (int k = 0; k < HELD_N; k++) {
      World w(500 + k), w2(500 + k);
      for (int ep = 0; ep < 40; ep++) {
        Vec o = w.reset(scenario(HELD_BASE + k)), o2 = w2.reset(scenario(HELD_BASE + k));
        if (o != o2) det = false;
        bool done = false;
        for (int t = 0; t < w.maxT() && !done; t++) {
          for (float v : o) if (!(v >= -1.f && v <= 1.f)) inrange = false;
          Transition tr = w.step(greedy_action(w)); o = tr.ns; done = tr.done;
        }
        if (!done) solv = false;
      }
      if (w.pz.nx.empty() || w.pz.n < 20) shown = false;
    }
    CHECK(solv, "held-out family: the BFS oracle solves all 20 puzzles in every episode");
    CHECK(inrange && det && shown, "held-out family: observations in [-1,1], deterministic reset, non-trivial state spaces");
  }
  {  // external environment: protocol, failure ends the episode without counting as a goal
    const std::string keep = g_ext_cmd; auto keep_p = g_ext;
    write_text_file("/tmp/nexus_selftest_bridge.py", BRIDGE_PY);
    g_ext_cmd = "python3 /tmp/nexus_selftest_bridge.py fake:CartPole 3"; g_ext.reset();
    World w(1); Vec o = w.reset(scenario(GAME_BASE + 200));
    bool inr = true; for (float v : o) if (!(v >= -1.f && v <= 1.f)) inr = false;
    bool failed = false, wrongdone = false, lastterm = false; float lastr = 0; int steps = 0;
    for (int t = 0; t < w.maxT(); t++) { Transition tr = w.step(1); steps++; lastr = tr.r; lastterm = tr.term; if (tr.done) wrongdone = true; if (w.maxT() == 0) { failed = true; break; } }
    CHECK(inr && w.pz.nx.empty(), "external env: reset gives an observation in [-1,1]");
    CHECK(failed && !wrongdone && steps > 3 && steps < 100 && lastterm && std::abs(lastr + 1.01f) < 1e-4f, "external env: pushing one way makes the pole fall, the episode ends with reward -1.01, flagged terminal, no goal");
    g_ext.reset(); g_ext_cmd = keep; g_ext = keep_p;
  }
  std::cout << (fails ? "SELFTEST FAILED: " : "SELFTEST PASSED: ") << (checks - fails) << "/" << checks << " checks\n";
  return fails;
}

}  // namespace nx

int main(int argc, char** argv) {
  using namespace nx;
  int episodes = 1200, evalEvery = 100, evalEps = 10, finalEps = 20;
  std::string ckpt = "nexus_cognitive.ckpt", csvname = "nexus_cognitive_metrics.csv";
  bool resume = false, doSelftest = false, evalOnly = false, presetBest = false, presetStable = false, presetGame = false, episodesSet = false, finalSet = false;
  uint32_t seed = 20261029;
  try {
    for (int i = 1; i < argc; i++) {
      std::string a = argv[i];
      auto next = [&]() -> std::string {
        if (i + 1 >= argc) throw std::runtime_error("missing value after " + a);
        return argv[++i];
      };
      if (a == "--episodes") { episodes = std::max(1, std::stoi(next())); episodesSet = true; }
      else if (a == "--eval-every") evalEvery = std::max(1, std::stoi(next()));
      else if (a == "--eval-eps") evalEps = std::max(1, std::stoi(next()));
      else if (a == "--final-eps") { finalEps = std::max(0, std::stoi(next())); finalSet = true; }
      else if (a == "--resume") resume = true;
      else if (a == "--ext") g_ext_cmd = next();
      else if (a == "--write-bridge") { write_text_file(next(), BRIDGE_PY); std::exit(0); }
      else if (a == "--write-llm-hints") { write_text_file(next(), LLM_HINTS_PY); std::exit(0); }
      else if (a == "--eval-only") { evalOnly = true; resume = true; }
      else if (a == "--no-heuristic") cfg.heur = false;
      else if (a == "--sparse") { cfg.sparse = true; cfg.heur = false; }
      else if (a == "--no-label") cfg.label = false;
      else if (a == "--no-ctx") cfg.ctx = false;
      else if (a == "--oracle-reward") cfg.oracle = true;
      else if (a == "--value") { cfg.value = true; cfg.heur = false; }
      else if (a == "--preset") {
        const std::string name = next();
        if (name != "best" && name != "stable" && name != "mountaincar") throw std::runtime_error("unknown preset (available: best, stable, mountaincar)");
        presetBest = true; presetStable = name == "stable"; presetGame = name == "mountaincar";
        cfg.value = true; cfg.heur = false; cfg.label = false;
        if (presetGame) { cfg.horizon = 20; cfg.gamma = .99f; cfg.rnd_beta = 1.f; }  // later explicit flags override these cfg.value = true; cfg.heur = false; cfg.label = false;
      }
      else if (a == "--randomize") cfg.randomize = true;
      else if (a == "--suite") cfg.suite = true;
      else if (a == "--hidden") cfg.hidden = true;
      else if (a == "--final-hidden") cfg.final_hidden = true;
      else if (a == "--keydoor") cfg.keydoor = true;
      else if (a == "--rnet") cfg.rnet = true;
      else if (a == "--ms-diag") cfg.ms_diag = true;
      else if (a == "--show-wind") cfg.show_wind = true;
      else if (a == "--show-phase") cfg.show_phase = true;
      else if (a == "--lr-decay") { cfg.lr_from = std::stoi(next()); cfg.lr_to = std::stoi(next()); }
      else if (a == "--val-start") cfg.val_start = true;
      else if (a == "--tok-diag") cfg.tok_diag = true;
      else if (a == "--rw-diag") cfg.rw_diag = true;
      else if (a == "--curious") cfg.curious = std::max(0.f, std::stof(next()));
      else if (a == "--blind") cfg.blind = true;
      else if (a == "--graph") cfg.graph = std::min(512, std::max(0, std::stoi(next())));
      else if (a == "--graph-w") cfg.graph_w = std::stof(next());
      else if (a == "--rcap") cfg.rcap = std::min(REPLAY_CAP, std::max(1000, std::stoi(next())));
      else if (a == "--hier") cfg.hier = std::min(512, std::max(0, std::stoi(next())));
      else if (a == "--hier-T") cfg.hier_T = std::max(2, std::stoi(next()));
      else if (a == "--hier-w") cfg.hier_w = std::stof(next());
      else if (a == "--bottleneck") cfg.bottleneck = true;
      else if (a == "--synth") cfg.synth = true;
      else if (a == "--synth-probe") { cfg.synth_probe = std::stoi(next()); cfg.synth = true; cfg.synth_lib = true; cfg.synth_diag = true; }
      else if (a == "--auto") { cfg.autocfg = true; cfg.value = true; cfg.heur = false; cfg.label = false; cfg.synth = true; cfg.synth_lib = true; cfg.horizon = 20; cfg.gamma = .99f; }
      else if (a == "--synth-lib") { cfg.synth = true; cfg.synth_lib = true; }
      else if (a == "--synth-diag") { cfg.synth = true; cfg.synth_lib = true; cfg.synth_diag = true; }
      else if (a == "--rules") { cfg.rules = next(); cfg.synth = true; }
      else if (a == "--synth-report") { cfg.synth_report = next(); cfg.synth = true; cfg.synth_lib = true; }
      else if (a == "--vq") cfg.vq = std::max(0, std::stoi(next()));
      else if (a == "--best-from") cfg.best_from = std::stoi(next());
      else if (a == "--avg") cfg.avg = std::stof(next());
      else if (a == "--best-every") cfg.best_every = std::max(0, std::stoi(next()));
      else if (a == "--best-eps") cfg.best_eps = std::max(1, std::stoi(next()));
      else if (a == "--unroll") cfg.unroll = std::max(0, std::min(8, std::stoi(next())));
      else if (a == "--unroll-batch") cfg.unroll_batch = std::max(1, std::stoi(next()));
      else if (a == "--unroll-w") cfg.unroll_w = std::stof(next());
      else if (a == "--train-ids" || a == "--final-ids" || a == "--val-ids") {
        std::vector<int>& dst = a == "--train-ids" ? cfg.train_ids : a == "--val-ids" ? cfg.val_ids : cfg.final_ids;
        std::stringstream ss(next());
        std::string tok;
        auto ok = [](int v) { return (v >= 0 && v < N_SCN) || (v >= GEN_BASE && v < GEN_BASE + GEN_MAX) || (v >= GAME_BASE && v <= GAME_BASE + 4) || (v >= HELD_BASE && v < HELD_BASE + HELD_N) || v == GAME_BASE + 200; };
        while (std::getline(ss, tok, ',')) {  // items are ids or inclusive ranges a-b (e.g. 100-149 = generated structures 0..49)
          const size_t dash = tok.find('-');
          const int lo = std::stoi(tok.substr(0, dash)), hi = dash == std::string::npos ? lo : std::stoi(tok.substr(dash + 1));
          if (hi < lo || !ok(lo) || !ok(hi)) throw std::runtime_error("scenario id out of range");
          for (int v = lo; v <= hi; v++) dst.push_back(v);
        }
      }
      else if (a == "--clock-fix") cfg.clock_fix = true;
      else if (a == "--repeat") cfg.repeat = std::max(1, std::stoi(next()));
      else if (a == "--trace") cfg.trace = std::stoi(next());
      else if (a == "--wm-extra") cfg.wm_extra = std::stoi(next());
      else if (a == "--oracle-value") cfg.oracle_value = true;
      else if (a == "--rw" && i + 1 < argc) cfg.rw = std::stof(argv[++i]);
      else if (a == "--final-keydoor") cfg.final_keydoor = true;
      else if (a == "--goal-memory") cfg.gmem = true;
      else if (a == "--reservoir") cfg.reservoir = true;
      else if (a == "--rnd") cfg.rnd_beta = std::max(0.f, std::stof(next()));
      else if (a == "--cover") cfg.cover = std::max(0.f, std::stof(next()));
      else if (a == "--pos-frac") cfg.pos_frac = clampf(std::stof(next()), 0.f, .9f);
      else if (a == "--gamma") cfg.gamma = clampf(std::stof(next()), .5f, .999f);
      else if (a == "--horizon") cfg.horizon = std::min(MAXH, std::max(1, std::stoi(next())));
      else if (a == "--pop") cfg.pop = std::min(MAXPOP, std::max(8, std::stoi(next())));
      else if (a == "--iters") cfg.iters = std::min(8, std::max(1, std::stoi(next())));
      else if (a == "--q-updates") cfg.q_updates = std::min(8, std::max(1, std::stoi(next())));
      else if (a == "--checkpoint") ckpt = next();
      else if (a == "--csv") csvname = next();
      else if (a == "--seed") seed = (uint32_t)std::stoul(next());
      else if (a == "--selftest") doSelftest = true;
      else if (a == "--help") {
        std::cout << "NEXUS COGNITIVE ENGINE v8\n"
                     "  --episodes N  --eval-every N  --eval-eps N  --final-eps N  --seed N\n"
                     "  --resume  --checkpoint F  --csv F  --selftest\n  --preset stable (= best + 1500 episodes, lr decay, weight averaging, snapshot choice on separate structures)\n  --lr-decay A B  --avg TAU  --val-ids L  --best-every N  --best-eps N  --best-from E  --val-start\n  --preset best   (dense reward + learned value, 50 generated training structures, 1000 episodes, held-out test on 20 structures)\n"
                     "  --eval-only       frozen evaluation of a checkpoint (e.g. with a longer --horizon at test time)\n"
                     "  reward / prior:  --sparse  --no-heuristic  --oracle-reward (diagnostic)\n"
                     "  exploration:     --rnd BETA  --pos-frac F  --cover BETA (episodic search-memory bonus)\n"
                     "  planning:        --value  --horizon H  --pop P  --iters I  --gamma G  --q-updates N\n"
                     "  generalization:  --randomize  --no-label  --no-ctx  --suite  --hidden (train on hidden-goal search)  --goal-memory  --reservoir (learned-readout memory)  --keydoor (key-door task)\n";
        return 0;
      } else throw std::runtime_error("unknown option " + a);
    }
  } catch (const std::exception& e) {
    std::cerr << "Argument error: " << e.what() << "\n";
    return 2;
  }
  if (presetGame) {  // MountainCar-v0 (scenario 3000): learn from scratch with the game's own reward, frozen test on 50 episodes
    if (cfg.train_ids.empty()) cfg.train_ids = {GAME_BASE};
    if (cfg.final_ids.empty()) cfg.final_ids = {GAME_BASE};
    if (!episodesSet) { episodes = 600; episodesSet = true; }  // set the flags so that the generic 'best' defaults below do not overwrite them
    if (!finalSet) { finalEps = 50; finalSet = true; }
  }
  if (presetBest) {  // defaults of the best known setting; every explicit flag still wins
    if (cfg.train_ids.empty()) for (int v = GEN_BASE; v < GEN_BASE + 50; v++) cfg.train_ids.push_back(v);
    const bool finalIdsDefault = cfg.final_ids.empty();
    if (finalIdsDefault) for (int v = GEN_BASE + 1000; v < GEN_BASE + 1020; v++) cfg.final_ids.push_back(v);
    if (!episodesSet) episodes = presetStable ? 1500 : 1000;
    if (!finalSet) finalEps = presetStable ? 4 : 6;
    if (presetStable) {  // 'best' + the stabilisers measured in the notes; explicit flags still win
      if (finalIdsDefault) { cfg.final_ids.clear(); for (int v = GEN_BASE + 1000; v < GEN_BASE + 1060; v++) cfg.final_ids.push_back(v); }
      if (cfg.lr_to <= cfg.lr_from) { cfg.lr_from = 1000; cfg.lr_to = 1500; }
      if (cfg.avg <= 0.f) cfg.avg = 1e-4f;
      if (cfg.val_ids.empty()) for (int v = GEN_BASE + 1100; v < GEN_BASE + 1140; v++) cfg.val_ids.push_back(v);
      if (cfg.best_every == 0) { cfg.best_every = 100; cfg.best_from = 1000; }
      cfg.val_start = true;
    }
  }
  if (doSelftest) return selftest() ? 1 : 0;

  if (cfg.autocfg) {  // decide from a random-move probe of the training tasks (deterministic in the seed, so a resumed run repeats the decision)
    std::vector<int> ids = cfg.train_ids;
    if (ids.empty()) for (int v = GEN_BASE; v < GEN_BASE + 50; v++) ids.push_back(v);
    std::mt19937 G(seed + 99);
    World pw(seed + 98);
    std::array<std::unordered_map<int, int>, OBS> seen;
    std::array<float, OBS> lo, hi; lo.fill(1e9f); hi.fill(-1e9f);
    int n = 0, goals = 0, eps = 0, rn = 0, fails = 0; double rsum = 0, rsq = 0; std::vector<float> rs; double dsum[OBS] = {}, dsq[OBS] = {};
    while (n < 3000 && eps < 600) {
      pw.reset(scenario(ids[G() % ids.size()])); eps++;
      for (int t = 0; t < pw.maxT() && n < 3000; t++) {
        Transition tr = pw.step((int)(G() % ACT)); n++;
        for (int j = 0; j < OBS; j++) { seen[j][(int)std::lround(tr.ns[j] * 1e4f)] = 1; lo[j] = std::min(lo[j], tr.ns[j]); hi[j] = std::max(hi[j], tr.ns[j]); }
        if (tr.done) { goals++; break; }
        if (tr.term) fails++;
        rn++; rsum += tr.r; rsq += (double)tr.r * tr.r; rs.push_back(tr.r);
        for (int j = 0; j < OBS; j++) { const double dj = (double)tr.ns[j] - (double)tr.s[j]; dsum[j] += dj; dsq[j] += dj * dj; }
      }
    }
    size_t maxd = 0; int varying = 0;
    for (int j = 0; j < OBS; j++) if (hi[j] - lo[j] > 1e-4f) { varying++; maxd = std::max(maxd, seen[j].size()); }
    const bool discrete = varying > 0 && maxd <= 600;
    const double rstd = rn > 10 ? std::sqrt(std::max(0.0, rsq / rn - (rsum / rn) * (rsum / rn))) : 0.0;
    // dense = the reward itself differs from step to step (a shaping signal). A reward that is constant except for rare events (a fall, a goal) has a
    // large std too but is NOT dense: judged by std alone, CartPole got horizon 8 / gamma .94 and could not see a fall 20 steps ahead (v20).
    double frac_off = 0;
    if (rs.size() > 10) {
      std::vector<float> tmp = rs; std::nth_element(tmp.begin(), tmp.begin() + (long)tmp.size() / 2, tmp.end());
      const float med = tmp[tmp.size() / 2];
      for (float v : rs) if (std::abs(v - med) > .005f) frac_off += 1.0 / (double)rs.size();
    }
    const bool dense = rstd > .02 && frac_off > .3;
    // step size per observation dimension: the world model predicts (next - current) / DSCALE. The defaults suit the arena; when one step changes a
    // dimension far less than that (CartPole: ~.01 per step against a scale of 1) the target is nearly zero and the network cannot beat 'nothing changes'. (v20)
    std::string scaled;
    if (rn > 50) for (int j = 0; j < OBS; j++) {
      const double mu = dsum[j] / rn, sd = std::sqrt(std::max(0.0, dsq[j] / rn - mu * mu));
      if (sd < 1e-6) continue;
      const float want = std::max(.003f, (float)(3.0 * sd));
      if (want < .25f * DSCALE[j]) { DSCALE[j] = want; scaled += " d" + std::to_string(j) + "=" + std::to_string(want).substr(0, 6); }
    }  // step rewards carry information of their own (shaping): a short horizon suffices and no novelty bonus is needed
    // a world with failures (episodes that end badly) is one where novelty is dangerous, not useful: the novelty bonus pulled CartPole to its edges (v21)
    const bool sparse = goals == 0 && !dense && fails == 0;
    if (dense) { cfg.horizon = 8; cfg.gamma = .94f; }
    if (discrete && cfg.vq == 0) cfg.vq = 256;
    if (sparse && cfg.rnd_beta == 0.f) cfg.rnd_beta = 1.f;
    std::cout << "auto: probe " << n << " random moves in " << eps << " episodes, goal reached " << goals << "x, most distinct values in one dimension " << maxd
              << " -> " << (discrete ? "discrete world: symbols (vq 256) ON" : "continuous world: symbols off") << ", " << (sparse ? "sparse goal: novelty bonus ON" : "novelty bonus off") << ", step-reward std " << rstd << ", share of steps off the usual value " << frac_off << " -> " << (dense ? "dense reward: horizon 8, gamma .94" : "sparse reward: horizon 20, gamma .99") << (scaled.empty() ? "" : ", step-size scales:" + scaled) << std::endl;
  }
  Agent agent(seed);
  if (!cfg.rules.empty() && !agent.sy.load_rules(cfg.rules)) { std::cerr << "cannot read rules file " << cfg.rules << "\n"; return 2; }
  World world(seed + 77);
  std::mt19937 task_rng(seed ^ 0x9e3779b9u);

  if (resume) {
    if (agent.load(ckpt)) std::cout << "Loaded " << ckpt << " (episodes=" << agent.episodes_done << ", steps=" << agent.steps << ", replay=" << agent.replay.size() << ")\n";
    else { std::cout << "Checkpoint missing/incompatible -> fresh run.\n"; resume = false; }
  }
  if (cfg.synth && resume) agent.sy.fit(agent.replay);
  if (cfg.synth_probe > 0) {
    std::vector<Transition> data; std::mt19937 G(seed + 5);
    World pw(seed + 11);
    for (int ep = 0; data.size() < 12000 && ep < 4000; ep++) {
      pw.reset(scenario(cfg.synth_probe));
      for (int t = 0; t < pw.maxT(); t++) { Transition tr = pw.step((int)(G() % ACT)); data.push_back(tr); if (tr.done) break; }
    }
    agent.sy.fit(data);
    std::cout << "probe scenario " << cfg.synth_probe << ": " << data.size() << " random-policy transitions\n" << agent.sy.summary() << "\n";
    for (const auto& v : agent.sy.verdicts) std::cout << "  " << v << "\n";
    if (!cfg.synth_report.empty()) agent.sy.write_report(cfg.synth_report, data);
    for (int a = 0; a < ACT; a++) for (int j = 0; j < OBS; j++) {
      const auto& u = agent.sy.r[a][j];
      if (u.ok && (u.lib || u.q > 0.f)) std::cout << "  action " << a << ": " << agent.sy.describe(a, j) << "\n";
      else if (!u.ok && u.acc > 0.f) std::cout << "  action " << a << " dim " << j << ": no rule (best held-out fit " << (int)(100 * u.acc) << "%)\n";
    }
    return 0;
  }
  if (cfg.synth_diag && resume) {
    std::cout << agent.sy.summary() << "\n";
    for (const auto& v : agent.sy.verdicts) std::cout << "  " << v << "\n";
    if (!cfg.synth_report.empty()) agent.sy.write_report(cfg.synth_report, agent.replay);
    for (int a = 0; a < ACT; a++) for (int j = 0; j < OBS; j++) {
      const auto& u = agent.sy.r[a][j];
      if (u.ok && (u.lib || u.q > 0.f)) std::cout << "  action " << a << ": " << agent.sy.describe(a, j) << "\n";
      else if (!u.ok && u.acc > 0.f) std::cout << "  action " << a << " dim " << j << ": no rule (best held-out fit " << (int)(100 * u.acc) << "%)\n";
    }
    return 0;
  }
  if (evalOnly) {  // no training: frozen evaluation of a saved agent, optionally with other planner settings
    if (!resume) { std::cerr << "--eval-only needs a valid checkpoint (--checkpoint F)\n"; return 2; }
    episodes = 0;
    for (int k = 0; k < cfg.wm_extra; k++) agent.learn_world();  // diagnostic: extra world-model training on the stored replay
    if (cfg.trace >= 0) { trace_diag(agent, seed, cfg.trace, finalEps > 0 ? finalEps : 20); return 0; }
    if (cfg.rw_diag) { for (int id : eval_ids()) rw_diag(agent, seed, id, finalEps > 0 ? finalEps : 30); return 0; }
    if (cfg.tok_diag) { for (int id : eval_ids()) tok_diag(agent, seed, id, finalEps > 0 ? finalEps : 50); return 0; }
    if (cfg.ms_diag) { for (int id : eval_ids()) multistep_diag(agent, seed, id, finalEps > 0 ? finalEps : 20); return 0; }
  }

  const std::string header = "episode,train_success_50,train_reward_50,train_steps_50,cum_goals,eval_t0_success,eval_t0_reward,eval_t1_success,eval_t1_reward,eval_t2_success,eval_t2_reward,eval_mean_success,train_loss,q_loss,eval_model_mse,eval_persist_mse,pos_in_replay,rhead_pos,rhead_neg,q_mean,q_at_goal,rnd_ema,replay,gradient_updates,seed,seconds";
  {  // never mix old and new column layouts in one file
    std::error_code ec;
    if (resume && std::filesystem::exists(csvname, ec)) {
      std::ifstream in(csvname);
      std::string first;
      std::getline(in, first);
      in.close();
      if (first != header) std::filesystem::rename(csvname, csvname + ".old", ec);
    }
  }
  std::ofstream csv(csvname, resume ? std::ios::app : std::ios::trunc);
  if (!csv) { std::cerr << "Cannot write metrics CSV\n"; return 2; }
  if (csv.tellp() <= 0) csv << header << "\n";

  std::cout << "NEXUS v13 | " << cfg_string() << " | ensemble=" << ENSEMBLE << " | seed=" << seed << std::endl;
  {  // reference points
    std::mt19937 br(seed + 5);
    std::cout << std::fixed << std::setprecision(1) << "baselines (success %) random / greedy-oracle:";
    for (int t : eval_ids()) {
      double rs = 0, gs = 0;
      World bw(seed + 4242 + (uint32_t)t * 1000);
      Scn sc = scenario(t);
      for (int j = 0; j < 20; j++) { rs += run_baseline(bw, sc, false, br).success; gs += run_baseline(bw, sc, true, br).success; }
      std::cout << "  " << sc.name << " " << rs * 5 << "/" << gs * 5;
    }
    std::cout << std::endl;
  }

  auto t0 = std::chrono::steady_clock::now();
  const std::string best_path = ckpt + ".best";
  double best_val = -1.0;
  int best_ep = 0;
  if (cfg.best_every > 0 && cfg.val_start && agent.episodes_done > 0) {  // --val-start: the snapshot we start from is a candidate too
    const double v0 = validate(agent, seed, cfg.best_eps, agent.episodes_done);
    std::cout << std::fixed << std::setprecision(3) << "  validation of the starting snapshot (episode " << agent.episodes_done << "): " << v0 << std::endl;
    agent.avg_swap(); const bool ok0 = agent.save(best_path); agent.avg_swap();
    if (ok0) { best_val = v0; best_ep = (int)agent.episodes_done; }
  }
  std::vector<double> sr, rr, ss;
  for (int k = 1; k <= episodes; k++) {
    uint64_t gep = agent.episodes_done + 1;  // global episode index (survives --resume)
    Scn sc = cfg.randomize ? Scn{"DR", -1, random_dyn(world.rng), MapMode::Random} : (!cfg.train_ids.empty() ? scenario(cfg.train_ids[task_rng() % cfg.train_ids.size()]) : scenario((int)(task_rng() % N_SEEN) + (cfg.hidden ? 9 : cfg.keydoor ? 12 : 0)));
    Stats z = run_episode(agent, world, sc, gep, true);
    agent.episodes_done++;
    agent.goals += (uint64_t)z.success;
    sr.push_back(z.success); rr.push_back(z.reward); ss.push_back(z.steps);
    if (sr.size() > 50) { sr.erase(sr.begin()); rr.erase(rr.begin()); ss.erase(ss.begin()); }

    if (k % evalEvery == 0 || k == episodes) {
      auto mean = [](const std::vector<double>& v) { return std::accumulate(v.begin(), v.end(), 0.0) / v.size(); };
      ScnResult res[N_SEEN];
      double merr = 0, perr = 0;
      // v20: the periodic test runs on the evaluation ids (--final-ids), not always on the three hand-made arena scenarios 0..2 (which said nothing
      // about a game being trained and cost two thirds of the test time when only one id is evaluated); missing slots repeat the first result
      const std::vector<int> evs = eval_ids();
      const int ne = (int)std::min<size_t>((size_t)N_SEEN, evs.size());
      for (int tid = 0; tid < N_SEEN; tid++) {
        res[tid] = tid < ne ? eval_scenario(agent, seed, evs[(size_t)tid], evalEps, gep) : res[0];
        merr += res[tid].merr / N_SEEN; perr += res[tid].perr / N_SEEN;
      }
      Diag dg = agent.diagnostics();
      double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      double ms = (res[0].succ + res[1].succ + res[2].succ) / 3;
      std::cout << std::fixed << "ep=" << std::setw(5) << gep << " | train succ=" << std::setprecision(1) << mean(sr) * 100
                << "% goals=" << agent.goals << " | eval succ T0/T1/T2=" << res[0].succ * 100 << "/" << res[1].succ * 100 << "/" << res[2].succ * 100
                << " (mean " << ms * 100 << "%) | mse model/persist=" << std::setprecision(5) << merr << "/" << perr
                << " | rhead pos/neg=" << std::setprecision(2) << dg.rpos << "/" << dg.rneg << " Q=" << dg.qmean << " | loss=" << std::setprecision(4) << agent.last_loss
                << " | " << std::setprecision(0) << secs << "s" << std::endl;
      csv << std::fixed << std::setprecision(6) << gep << "," << mean(sr) << "," << mean(rr) << "," << mean(ss) << "," << agent.goals;
      for (int t = 0; t < N_SEEN; t++) csv << "," << res[t].succ << "," << res[t].rew;
      csv << "," << ms << "," << agent.last_loss << "," << agent.last_qloss << "," << merr << "," << perr << "," << dg.pos << ","
          << dg.rpos << "," << dg.rneg << "," << dg.qmean << "," << dg.qgoal << "," << agent.rnd_ema << "," << agent.replay.size() << ","
          << agent.updates << "," << seed << "," << secs << "\n";
      csv.flush();
      if (!agent.save(ckpt)) std::cerr << "WARNING: checkpoint save failed\n";
    }
    if (cfg.best_every > 0 && (int)gep >= cfg.best_from && (k % cfg.best_every == 0 || k == episodes)) {  // validation + best-snapshot bookkeeping
      const double v = validate(agent, seed, cfg.best_eps, gep);
      const bool better = v >= best_val;  // ties favour the later (more trained) snapshot
      std::cout << std::fixed << std::setprecision(3) << "  validation @ep " << gep << ": " << v << (better ? "  <- best so far" : "") << std::endl;
      if (better) { agent.avg_swap(); const bool ok = agent.save(best_path); agent.avg_swap(); if (ok) { best_val = v; best_ep = (int)gep; } }
    }
  }
  if (cfg.best_every > 0 && best_ep > 0) {
    const double last_val = validate(agent, seed, cfg.best_eps, agent.episodes_done);
    if (best_ep != (int)agent.episodes_done || cfg.avg > 0.f) {  // with --avg the saved snapshot holds averaged weights, so it is always restored
      if (agent.load(best_path)) {
        agent.avg_ready = false;
        const double again = validate(agent, seed, cfg.best_eps, agent.episodes_done);
        std::cout << std::fixed << std::setprecision(3) << "BEST snapshot restored: episode " << best_ep << " (validation " << best_val << ", last snapshot " << last_val
                  << ", re-validated after restore " << again << (std::abs(again - best_val) < 1e-9 ? ", identical" : ", MISMATCH!") << ")" << std::endl;
      } else std::cerr << "WARNING: could not restore the best snapshot; evaluating the last one\n";
    } else std::cout << "BEST snapshot is the last one (episode " << best_ep << ", validation " << best_val << ")" << std::endl;
  }

  if (cfg.avg > 0.f && !(cfg.best_every > 0 && best_ep > 0)) agent.avg_swap();  // final evaluation with the averaged weights
  if (finalEps > 0) {  // larger frozen evaluation (and the transfer suite with --suite)
    const std::vector<int> ids = eval_ids();
    std::ofstream fin(csvname + ".final.csv", std::ios::app);
    if (fin.tellp() <= 0) fin << "seed,config,scenario,success,reward,steps,n,model_mse,persist_mse,found,succ_if_found\n";
    double tot = 0, tot_all = 0; int n_seen = 0;
    std::cout << std::fixed << std::setprecision(1) << "FINAL (frozen agent, " << finalEps << " episodes per scenario):\n";
    for (int id : ids) {
      ScnResult r = eval_scenario(agent, seed, id, finalEps, agent.episodes_done);
      Scn sc = scenario(id);
      std::cout << "  " << std::left << std::setw(11) << sc.name << std::right << " success " << std::setw(5) << r.succ * 100 << "%  reward "
                << std::setprecision(2) << r.rew << std::setprecision(1) << "  steps " << r.steps << "  goal found " << r.found * 100 << "%  success once found " << r.succ_found * 100 << "%\n";
      fin << seed << "," << cfg_string() << "," << sc.name << "," << r.succ << "," << r.rew << "," << r.steps << "," << finalEps << ","
          << r.merr << "," << r.perr << "," << r.found << "," << r.succ_found << "\n";
      if (id < N_SEEN) { tot += r.succ / N_SEEN; n_seen++; }
      tot_all += r.succ;
    }
    // 'seen-mean' only exists for the hand-made seen scenarios; it used to print 0.0% for every other id set (games, generated structures), which reads like a failure
    if (!ids.empty()) std::cout << "FINAL mean success over all " << ids.size() << " scenarios: " << tot_all / (double)ids.size() * 100 << "%" << std::endl;
    if (n_seen > 0 && !cfg.final_hidden && !cfg.final_keydoor) std::cout << "FINAL seen-mean success " << tot * 100 << "%" << std::endl;
  }
  std::cout << "Done. transitions=" << agent.steps << ", goals reached in training=" << agent.goals << "\nCheckpoint: " << ckpt
            << (cfg.synth ? "\n" + agent.sy.summary() : std::string()) << "\nMetrics: " << csvname << "\nNote: research prototype, not general intelligence." << std::endl;
  return 0;
}
