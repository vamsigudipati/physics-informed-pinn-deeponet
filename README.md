# Physics-informed learning: PINNs, an inverse problem and a DeepONet

These five notebooks follow one thread through physics-informed machine learning on heat conduction and porous flow. In Stage 1 I solve one field by hand: a physics-informed neural network (PINN) in plain PyTorch, applied with the same code to a rod, a cylindrical fuel pin and a plate, each checked against an exact or finite-difference reference. Stage 2 does the same in the DeepXDE library and turns it around into an inverse problem, recovering a conductivity, a viscosity and a front coefficient from sparse measurements, including cases the data cannot identify. Stage 3 writes the inverse idea by hand in a few dozen lines, so the library version and the hand version can be compared. Stage 4 stops solving one field at a time and learns the whole family: a DeepONet that maps any permeability field to its Darcy pressure field in one forward pass. Stage 5 asks two questions across the earlier stages: how much training data the DeepONet needs, and how robust the two inverse solvers are to noisy measurements.

Portfolio page: https://vamsigudipati.github.io/physics-informed-pinn-deeponet.html

## Notebooks

| Notebook | Stage | Results | Models | Animations |
| --- | --- | --- | --- | --- |
| `notebooks/01_pinn_heat_conduction_from_scratch.ipynb` | Stage 1 — PINN for heat conduction, built from scratch | `results/01_pinn_heat_conduction_from_scratch/` | `models/01_pinn_heat_conduction_from_scratch/` | `animations/01_*` |
| `notebooks/02_deepxde_forward_and_inverse.ipynb` (with `test_pinn_inverse.py`) | Stage 2 — DeepXDE: forward and inverse problems | `results/02_deepxde_forward_and_inverse/` | `models/02_deepxde_forward_and_inverse/` | `animations/02_*` |
| `notebooks/03_inverse_pinn_by_hand.ipynb` | Stage 3 — An inverse PINN by hand | `results/03_inverse_pinn_by_hand/` | `models/03_inverse_pinn_by_hand/` | `animations/03_*` |
| `notebooks/04_deeponet_darcy_operator.ipynb` (with `deeponet_slide.py`) | Stage 4 — DeepONet: the Darcy flow operator | `results/04_deeponet_darcy_operator/` | `models/04_deeponet_darcy_operator/` | `animations/04_*` |
| `notebooks/05_data_efficiency_and_noise.ipynb` | Stage 5 — Data efficiency and noise robustness | `results/05_data_efficiency_and_noise/` | `models/05_data_efficiency_and_noise/` | — |

Each notebook is also exported as HTML next to it, with all outputs. In `results/`, the `exercises/` folder of each stage holds the tables (CSV) and figures (PNG) of the experiments and of the three-seed repeats; Stage 5 writes its tables and figures directly into its results folder. `models/` holds the trained networks as PyTorch state dictionaries (and DeepXDE checkpoints for Stage 2); `animations/` holds the training videos and their final frames.

## Setting up and running

I ran everything on a laptop CPU (Apple M3 Max) with Python 3.12, PyTorch 2.14 and DeepXDE 1.15 with its PyTorch backend. The frozen environment is in `requirements.txt`; the videos use the ffmpeg binary that `imageio-ffmpeg` installs.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name physics-informed
export DDE_BACKEND=pytorch
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 01_pinn_heat_conduction_from_scratch.ipynb
```

All paths inside the notebooks are relative to `notebooks/`, so run them from there; each notebook creates its own folders under `results/`, `models/` and `animations/`. Every model is trained when the notebook runs, and nothing is loaded from a previous run. On the laptop, with Stages 1, 2 and 4 running side by side and then Stages 3 and 5, Stage 1 took about 11 minutes, Stage 2 about 5, Stage 4 about 10, Stage 3 under a minute and Stage 5 about 3.

**Why the CPU.** Importing DeepXDE 1.15 makes the Apple GPU (MPS) the default PyTorch device for every tensor created afterwards. On this machine DeepXDE is slower there (4.4 ms per Adam iteration against 1.9 ms on the CPU), and the small hand-written networks gain nothing either, so each notebook sets `torch.set_default_device("cpu")` right after it imports DeepXDE.

**Why three seeds.** A single training run gives one draw from the initialisation and the random point sets. Wherever a result depends on them I repeat it with the seeds 0, 1 and 2 and report the mean and the standard deviation; a difference smaller than that spread is not an effect. Three of the original results turn out to be one favourable or unfavourable draw, and the seed tables show which.

## Data

There is no external data. Every reference is either a manufactured exact solution (the rod and the plate of Stage 1), an analytical solution (the channel flow and the Stefan front of Stage 2), or the notebook's own classical solver: finite differences for the fuel pin and the steady conduction problem, and a finite-volume Darcy solver that generates the training pairs of Stage 4.
