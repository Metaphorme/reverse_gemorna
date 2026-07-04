from pathlib import Path
from types import SimpleNamespace
import sys
import unittest


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gemorna_services import GemornaService, parse_generated_utr_output, validate_utr_length


class FakeCDSGenerator:
    def __init__(self):
        self.calls = []

    def translate_protein_to_rna(self, protein_sequence, seed=None):
        self.calls.append((protein_sequence, seed))
        return SimpleNamespace(
            protein_sequence=protein_sequence,
            dna_sequence="ATGGTT",
            rna_sequence="AUGGUU",
            naturalness=0.42,
            sampling_seed=17,
            device="cpu",
        )


class FakeUTRGenerator:
    def __init__(self):
        self.calls = []

    def generate(self, utr_type, length, seed=None):
        self.calls.append((utr_type, length, seed))
        return SimpleNamespace(
            utr_type=utr_type,
            length=length,
            sequence="ACGUAC",
            sampling_seed=31,
            device="cpu",
        )


class FakeUTRScorer:
    def __init__(self, score):
        self.score = score
        self.calls = []

    def score_sequence(self, sequence):
        self.calls.append(sequence)
        return SimpleNamespace(
            utr_type="fake",
            sequence=sequence,
            score=self.score,
            device="cpu",
        )


class GemornaServiceTests(unittest.TestCase):
    def test_validate_utr_length_accepts_expected_values(self):
        for value in ("short", "medium", "long"):
            self.assertEqual(validate_utr_length(value), value)

    def test_validate_utr_length_rejects_unknown_values(self):
        with self.assertRaisesRegex(ValueError, "short, medium, or long"):
            validate_utr_length("tiny")

    def test_parse_generated_utr_output_reads_last_sequence_line(self):
        output = "\nGenerated UTR:\nACGUAC\n"
        self.assertEqual(parse_generated_utr_output(output), "ACGUAC")

    def test_parse_generated_utr_output_rejects_missing_sequence(self):
        with self.assertRaisesRegex(ValueError, "Unable to parse"):
            parse_generated_utr_output("Generated UTR:\n")

    def test_generate_cds_open_uses_open_generator(self):
        generator = FakeCDSGenerator()
        service = GemornaService(open_cds_generator=generator)

        result = service.generate_cds_open("MV", seed=5)

        self.assertEqual(generator.calls, [("MV", 5)])
        self.assertEqual(result.implementation, "open")
        self.assertEqual(result.dna_sequence, "ATGGTT")
        self.assertEqual(result.rna_sequence, "AUGGUU")
        self.assertEqual(result.naturalness, 0.42)

    def test_generate_cds_closed_uses_closed_generator(self):
        generator = FakeCDSGenerator()
        service = GemornaService(closed_cds_generator=generator)

        result = service.generate_cds_closed("MV", seed=11)

        self.assertEqual(generator.calls, [("MV", 11)])
        self.assertEqual(result.implementation, "closed")
        self.assertEqual(result.sampling_seed, 17)

    def test_generate_utr_scores_generated_sequence(self):
        generator = FakeUTRGenerator()
        scorer = FakeUTRScorer(8.75)
        service = GemornaService(
            utr_generator=generator,
            utr_scorers={"5utr": scorer},
        )

        result = service.generate_utr("5utr", "short", seed=7)

        self.assertEqual(generator.calls, [("5utr", "short", 7)])
        self.assertEqual(scorer.calls, ["ACGUAC"])
        self.assertEqual(result.utr_type, "5utr")
        self.assertEqual(result.length, "short")
        self.assertEqual(result.sequence, "ACGUAC")
        self.assertEqual(result.score, 8.75)
        self.assertEqual(result.sampling_seed, 31)

    def test_score_utr_uses_matching_predictor(self):
        five_scorer = FakeUTRScorer(1.0)
        three_scorer = FakeUTRScorer(2.0)
        service = GemornaService(
            utr_scorers={
                "5utr": five_scorer,
                "3utr": three_scorer,
            }
        )

        result = service.score_utr("3utr", "ACGU")

        self.assertEqual(five_scorer.calls, [])
        self.assertEqual(three_scorer.calls, ["ACGU"])
        self.assertEqual(result.utr_type, "3utr")
        self.assertEqual(result.sequence, "ACGU")
        self.assertEqual(result.score, 2.0)


if __name__ == "__main__":
    unittest.main()
