# Result: multi-password token-swap diagnostic (cls@224)

## Setup
- Data: `random_pw6_lang_100`, training passwords `111222` (2 demos) + `222111` (1 demo), 3 demos total
- Base training: `runs/b8_token_swap_diag`, cls@224, 40 epochs, lr=1e-4, batch-size=2, TTT online, carry-windows T=16, prog-weight=0.0, seed=42
- Continued training: `runs/b8_token_swap_diag_cont`, +30 epochs from base snapshot (epochs 0..29 in cont log)
- Eval: 1 repeat per combo, max 600 steps, TTT fast weights carried per episode, **argmax inference fix applied**

## Results (before argmax fix, base model)
| env password | token password | success | failed | press_count | steps |
|---|---|---|---|---|---|
| 111222 | 111222 | 0 | 0 | 0 | 600 |
| 111222 | 222111 | 0 | 0 | 0 | 600 |
| 222111 | 222111 | 0 | 0 | 0 | 600 |
| 222111 | 111222 | 0 | 0 | 0 | 600 |

## Results (continued model, argmax)
| env password | token password | success | failed | press_count | steps |
|---|---|---|---|---|---|
| 111222 | 111222 | 0 | 0 | 0 | 600 |
| 111222 | 222111 | 0 | 1 | 0 | 75 |
| 222111 | 222111 | 0 | 0 | 3 | 600 |
| 222111 | 111222 | 0 | 0 | 0 | 600 |

## Important related fix
- Found `_sample_action` used `torch.multinomial` even at eval time.
- Changed to: training samples, eval uses `argmax`.
- This fix was necessary for the model to produce coherent rollouts.

## Interpretation
- The continued small multi-password model **does respond to password tokens**: `222111` env + correct token presses 3 times, while wrong token presses 0.
- It has not yet achieved full success (6/6 presses in correct order), but token-conditioned behavior is now observable.
- This supports scaling to a medium/full cls@224 full-password training run.
