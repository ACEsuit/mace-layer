# MACE - Diffusion

Generative diffusion model based on MACE.

## Installation

Requirements:
* Python >= 3.7
* [PyTorch](https://pytorch.org/) >= 1.8

### conda installation

If you do not have CUDA pre-installed, it is **recommended** to follow the conda installation process:
```sh
# Create a virtual environment and activate it
conda create mace_env
conda activate mace_env

# Install PyTorch
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c conda-forge

# Clone and install MACE (and all required packages), use token if still private repo
git clone git@github.com:ACEsuit/mace.git 
pip install ./mace
```

### pip installation

To install via `pip`, follow the steps below:
```sh
# Create a virtual environment and activate it
python -m venv mace-venv
source mace-venv/bin/activate

# Install PyTorch (for example, for CUDA 10.2 [cu102])
pip install torch==1.8.2 --extra-index-url "https://download.pytorch.org/whl/lts/1.8/cu102"

# Clone and install MACE (and all required packages)
git clone git@github.com:ACEsuit/mace.git
pip install ./mace
```

**Note:** The homonymous package on [PyPI](https://pypi.org/project/MACE/) has nothing to do with this one.

## Usage

## Development

We use `black`, `isort`, `pylint`, and `mypy`.
Run the following to format and check your code:
```sh
bash ./scripts/run_checks.sh
```

We have CI set up to check this, but we _highly_ recommend that you run those commands
before you commit (and push) to avoid accidentally committing bad code.

We are happy to accept pull requests under an [MIT license](https://choosealicense.com/licenses/mit/). Please copy/paste the license text as a comment into your pull request.

## References

If you use this code, please cite our papers:
```text
@misc{Batatia2022MACE,
  title = {MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields},
  author = {Batatia, Ilyes and Kov{\'a}cs, D{\'a}vid P{\'e}ter and Simm, Gregor N. C. and Ortner, Christoph and Cs{\'a}nyi, G{\'a}bor},
  year = {2022},
  number = {arXiv:2206.07697},
  eprint = {2206.07697},
  eprinttype = {arxiv},
  doi = {10.48550/ARXIV.2206.07697},
  archiveprefix = {arXiv}
}
@misc{Batatia2022Design,
  title = {The Design Space of E(3)-Equivariant Atom-Centered Interatomic Potentials},
  author = {Batatia, Ilyes and Batzner, Simon and Kov{\'a}cs, D{\'a}vid P{\'e}ter and Musaelian, Albert and Simm, Gregor N. C. and Drautz, Ralf and Ortner, Christoph and Kozinsky, Boris and Cs{\'a}nyi, G{\'a}bor},
  year = {2022},
  number = {arXiv:2205.06643},
  eprint = {2205.06643},
  eprinttype = {arxiv},
  doi = {10.48550/arXiv.2205.06643},
  archiveprefix = {arXiv}
 }
```

## Contact

If you have any questions, please contact us at ilyes.batatia@ens-paris-saclay.fr.

For bugs or feature requests, please use [GitHub Issues](https://github.com/ACEsuit/mace/issues).

## License

MACE is published and distributed under the [Academic Software License v1.0 ](ASL.md).
