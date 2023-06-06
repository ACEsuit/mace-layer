import torch
from mace_layer import cg
from mace_layer.symmetric_contraction import SymmetricContraction

from e3nn import o3


class TestSymmetricContract:
    def test_symmetric_contraction(self):
        operation = SymmetricContraction(
            irreps_in=o3.Irreps("16x0e + 16x1o + 16x2e"),
            irreps_out=o3.Irreps("16x0e + 16x1o"),
            correlation=3,
            num_elements=2,
        )
        torch.manual_seed(123)
        features = torch.randn(30, 16, 9)
        one_hots = torch.nn.functional.one_hot(torch.arange(0, 30) % 2).to(
            torch.get_default_dtype()
        )
        out = operation(features, one_hots)
        assert out.shape == (30, 64)
        assert operation.contractions[0].weights_max.shape == (2, 11, 16)


def test_U_matrix():
    irreps_in = o3.Irreps("1x0e + 1x1o + 1x2e")
    irreps_out = o3.Irreps("1x0e + 1x1o")
    u_matrix = cg.U_matrix_real(
        irreps_in=irreps_in, irreps_out=irreps_out, correlation=3
    )[-1]
    assert u_matrix.shape == (3, 9, 9, 9, 21)
