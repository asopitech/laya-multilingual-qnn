import importlib.util
import sys
import unittest
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "demos" / "rlcd_finetune.py"
SPEC = importlib.util.spec_from_file_location("rlcd_finetune", MODULE_PATH)
demo = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = demo
SPEC.loader.exec_module(demo)


class RlcdFineTuneTest(unittest.TestCase):
    def test_one_hot(self):
        self.assertEqual([0.0, 1.0, 0.0], demo.one_hot(1, 3))

    def test_parameter_groups(self):
        self.assertEqual("encoder", demo.parameter_group("encoder.layer.weight"))
        self.assertEqual("act_head", demo.parameter_group("act_head.0.weight"))
        self.assertEqual("decision_head", demo.parameter_group("head.layers.0.weight"))
        self.assertEqual("decision_head", demo.parameter_group("scorer.1.weight"))

    def test_rlcd_loss_reaches_logits(self):
        torch.manual_seed(3)
        logits = torch.tensor([[0.2, -0.1, 0.5]], requires_grad=True)
        batch = {
            "marker_mask": torch.tensor([[True, True, True]]),
            "target": torch.tensor([[0.0, 1.0, 0.0]]),
            "qtype": torch.tensor([0]),
        }
        loss, metrics = demo.rlcd_loss(logits, batch, group_size=4, sigma=0.2)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(logits.grad.norm()), 0.0)
        self.assertIn("reward", metrics)


if __name__ == "__main__":
    unittest.main()
