from src.communication import CommunicationSimulator


def test_rejection_topk_payload_uses_values_and_indices() -> None:
    # Given: one float16 probability row compressed to top four
    # When: its canonical top-k payload size is calculated
    payload_bytes = CommunicationSimulator._compressed_topk_payload_bytes(
        compressed_k=4,
        seq_length=1,
        prob_element_size=2,
    )

    # Then: every transmitted value includes its int32 vocabulary index
    assert payload_bytes == 4 * (2 + 4)
