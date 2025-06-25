import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from event_weight import EventSignature, EventWeightCalculator


def test_signature_generation():
    sig = EventSignature()
    data = {
        "primary_actor": "Judge Patricia Dugan",
        "action": "taken into custody",
        "date": "April 1, 2025",
        "location": "Washington DC",
    }
    signature = sig.generate_signature(data)
    assert signature == "patricia_dugan|arrested|2025-04-01|washington_dc"


def test_weight_calculation():
    calc = EventWeightCalculator()
    observations = [
        {
            "source_bias": "left",
            "source_name": "A",
            "extracted_data": {"date": "2025-04-01"},
        },
        {
            "source_bias": "right",
            "source_name": "B",
            "extracted_data": {"date": "2025-04-01"},
        },
    ]
    metrics = calc.calculate_weight(observations)
    assert metrics["weight"] > 0
    assert metrics["source_diversity"] == 1.0

