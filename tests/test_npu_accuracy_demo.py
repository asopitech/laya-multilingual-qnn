import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "demos" / "npu_accuracy_demo.py"
SPEC = importlib.util.spec_from_file_location("npu_accuracy_demo", MODULE_PATH)
demo = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = demo
SPEC.loader.exec_module(demo)


class DemoHelpersTest(unittest.TestCase):
    def test_parse_law_options(self):
        parsed = demo.parse_law_options("a 最初の選択肢\nb 二番目\nの続き\nc 三番目\nd 四番目")
        self.assertEqual(["a", "b", "c", "d"], list(parsed))
        self.assertEqual("二番目 の続き", parsed["b"])

    def test_retrieval_prefers_matching_chunk(self):
        case = demo.DecisionCase(
            dataset="lawqa",
            uid="1",
            state="",
            question="届出の期限は何日ですか",
            kind="choice",
            criteria={"a": "十日", "b": "十五日"},
            gold_label="b",
            context="税率について定める。\n## 第8条\n届出は十五日を経過した日に効力を生ずる。",
        )
        self.assertIn("十五日", demo.retrieve_context(case, 1)[0])

    def test_context_chunks_honor_npu_sized_character_budget(self):
        chunks = demo.context_chunks("あ" * 250, max_chars=80)
        self.assertEqual(4, len(chunks))
        self.assertTrue(all(len(chunk) <= 80 for chunk in chunks))

    def test_temperature_scaling_preserves_argmax(self):
        original = {"a": 0.8, "b": 0.15, "c": 0.05}
        scaled = demo.scale_probabilities(original, 2.0)
        self.assertEqual("a", max(scaled, key=scaled.get))
        self.assertAlmostEqual(1.0, sum(scaled.values()))
        self.assertLess(scaled["a"], original["a"])

    def test_fit_temperature_softens_overconfident_errors(self):
        cases = [
            demo.DecisionCase("x", "1", "", "", "choice", {"a": "A", "b": "B"}, "b"),
            demo.DecisionCase("x", "2", "", "", "choice", {"a": "A", "b": "B"}, "a"),
        ]
        predictions = [
            demo.Prediction({"a": 0.99, "b": 0.01}, 1.0, 1, ["128x8"]),
            demo.Prediction({"a": 0.99, "b": 0.01}, 1.0, 1, ["128x8"]),
        ]
        self.assertGreater(demo.fit_temperature(cases, predictions), 1.0)


if __name__ == "__main__":
    unittest.main()
