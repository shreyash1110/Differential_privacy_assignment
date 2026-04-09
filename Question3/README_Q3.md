# Question 3: Effect of Gradient Clipping Norm in DP-SGD

## Objective
This experiment studies the effect of the gradient clipping norm $C$ on training dynamics and model utility by training the DP-SGD model with $C \in \{0.1, 10.0\}$. 

## Methodology
To isolate the impact of the clipping norm, the noise multiplier and batch size are kept constant between runs. To manage memory complexity during private training, this implementation utilizes flat clipping with ghost clipping support via Opacus, avoiding the explicit instantiation of full per-sample gradients.

## How to Run
To reproduce the clipping norm comparisons, run:
```bash
python q3.py
