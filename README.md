# mace-layer

A MACE layer for broad use on 3D point clouds.

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
git clone git@github.com:ACEsuit/mace-layer.git 
pip install ./mace-layer
```

### pip installation

To install via `pip`, follow the steps below:
```sh
# Create a virtual environment and activate it
python -m venv mace-venv
source mace-venv/bin/activate

# Install PyTorch (for example, for CUDA 11.6 [cu116])
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

# Clone and install MACE (and all required packages)
git clone git@github.com:ACEsuit/mace-layer.git
pip install ./mace-layer
```

**Note:** The homonymous package on [PyPI](https://pypi.org/project/MACE/) has nothing to do with this one.

## Usage

To create a mace layer use this code,

```python
from mace_layer import MACE_layer
layer = MACE_layer(
    max_ell=3,
    correlation=3,
    n_dims_in=2,
    hidden_irreps="16x0e + 16x1o + 16x2e",
    node_feats_irreps="16x0e + 16x1o + 16x2e",
    edge_feats_irreps="16x0e",
    avg_num_neighbors=10.0,
    use_sc=True,
)
node_feats = layer(
    vectors,
    node_feats,
    node_attrs,
    edge_feats,
    edge_index,
)
```
with the hyper parameters being,

```
        max_ell (int): Maximum angular momentum in the spherical expansion on edges, :math:`l = 0, 1, \dots`.
        Controls the resolution of the spherical expansion.
        correlation (int): The maximum correlation order of the messages, :math:`\nu = 0, 1, \dots`.
        n_dims_in (int): The number of input node attributes.
        hidden_irreps (str): The hidden irreps defining the node features to construct.
        node_feats_irreps (str): The irreps of the node features in the input.
        edge_feats_irreps (str): The irreps of the edge features in the input.
        avg_num_neighbors (float): A normalization factor for the pooling operation, 
        usually taken as the average number of neighbors.
        interaction_cls (Callable, optional): The type of interaction block to use. 
        Defaults to RealAgnosticResidualInteractionBlock.
        Defaults to False.
        use_sc (bool, optional): Whether to use the self connection. Defaults to True.
``` 

and the input,

```python
Shapes:
        - **input:**
            - **vectors** (torch.Tensor): The edge vectors of shape :math:`(|\mathcal{E}|, 3)`.
            - **node_feats** (torch.Tensor): The node features of shape :math:`(|\mathcal{V}|, \text{node\_feats\_irreps})`.
            - **node_attrs** (torch.Tensor): The node attributes of shape :math:`(|\mathcal{V}|, \text{n\_dims\_in})`.
            - **edge_feats** (torch.Tensor): The edge features of shape :math:`(|\mathcal{E}|, (\text{egde\_feats\_irreps}))`.
            - **edge_index** (torch.Tensor): The edge indices of shape :math:`(2, |\mathcal{E}|)`.
        - **output:**
            - **node_feats** (torch.Tensor): The node features of shape :math:`(|\mathcal{V}|, \text{hidden\_irreps})`.
```

## Development

We use `black`, `isort`, `pylint`, and `mypy`.
Run the following to format and check your code:
```sh
bash ./scripts/run_checks.sh
```

We have CI set up to check this, but we _highly_ recommend that you run those commands
before you commit (and push) to avoid accidentally committing bad code.


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

MACE is published and distributed under the [MIT license](LICENSE).
