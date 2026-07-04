from pathlib import Path
from types import SimpleNamespace
import sys
import unittest


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import api


class FakeService:
    def __init__(self):
        self.calls = []

    def generate_cds_open(self, protein_sequence, seed=None):
        self.calls.append(("cds_open", protein_sequence, seed))
        return SimpleNamespace(
            implementation="open",
            protein_sequence=protein_sequence,
            dna_sequence="ATG",
            rna_sequence="AUG",
            naturalness=0.5,
            sampling_seed=seed or 1,
            device="cpu",
        )

    def generate_cds_closed(self, protein_sequence, seed=None):
        self.calls.append(("cds_closed", protein_sequence, seed))
        return SimpleNamespace(
            implementation="closed",
            protein_sequence=protein_sequence,
            dna_sequence="ATG",
            rna_sequence="AUG",
            naturalness=0.4,
            sampling_seed=seed or 2,
            device="cpu",
        )

    def generate_utr(self, utr_type, length, seed=None):
        self.calls.append(("utr_generate", utr_type, length, seed))
        return SimpleNamespace(
            utr_type=utr_type,
            length=length,
            sequence="ACGU",
            score=7.25,
            sampling_seed=seed,
            device="cpu",
        )

    def score_utr(self, utr_type, sequence):
        self.calls.append(("utr_score", utr_type, sequence))
        return SimpleNamespace(
            utr_type=utr_type,
            sequence=sequence,
            score=6.5,
            device="cpu",
        )


class RaisingService(FakeService):
    def generate_cds_open(self, protein_sequence, seed=None):
        raise ValueError("bad protein")


class ApiContractTests(unittest.TestCase):
    def test_routes_are_registered(self):
        paths = {route.path for route in api.app.routes}

        self.assertIn("/health", paths)
        self.assertIn("/api/v1/cds/open/generate", paths)
        self.assertIn("/api/v1/cds/closed/generate", paths)
        self.assertIn("/api/v1/utr/5/generate", paths)
        self.assertIn("/api/v1/utr/3/generate", paths)
        self.assertIn("/api/v1/utr/5/score", paths)
        self.assertIn("/api/v1/utr/3/score", paths)

    def test_generate_cds_open_route_uses_service(self):
        service = FakeService()
        request = api.CDSGenerationRequest(protein_sequence="MV", seed=5)

        response = api.generate_cds_open(request, service=service)

        self.assertEqual(service.calls, [("cds_open", "MV", 5)])
        self.assertEqual(response.implementation, "open")
        self.assertEqual(response.rna_sequence, "AUG")

    def test_generate_cds_closed_route_uses_service(self):
        service = FakeService()
        request = api.CDSGenerationRequest(protein_sequence="MV", seed=8)

        response = api.generate_cds_closed(request, service=service)

        self.assertEqual(service.calls, [("cds_closed", "MV", 8)])
        self.assertEqual(response.implementation, "closed")

    def test_generate_utr_route_uses_service(self):
        service = FakeService()
        request = api.UTRGenerationRequest(length="short", seed=7)

        response = api.generate_5utr(request, service=service)

        self.assertEqual(service.calls, [("utr_generate", "5utr", "short", 7)])
        self.assertEqual(response.sequence, "ACGU")
        self.assertEqual(response.score, 7.25)

    def test_score_utr_route_uses_service(self):
        service = FakeService()
        request = api.UTRScoreRequest(sequence="ACGT")

        response = api.score_3utr(request, service=service)

        self.assertEqual(service.calls, [("utr_score", "3utr", "ACGT")])
        self.assertEqual(response.score, 6.5)

    def test_value_error_maps_to_http_400(self):
        request = api.CDSGenerationRequest(protein_sequence="BAD", seed=1)

        with self.assertRaises(api.HTTPException) as raised:
            api.generate_cds_open(request, service=RaisingService())

        self.assertEqual(raised.exception.status_code, 400)
        self.assertEqual(raised.exception.detail, "bad protein")


if __name__ == "__main__":
    unittest.main()
