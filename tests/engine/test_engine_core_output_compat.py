from vllm_omni.engine import OmniEngineCoreOutput


def test_engine_core_output_accepts_ec_transfer_params() -> None:
    transfer = {"connector": "test"}
    output = OmniEngineCoreOutput(
        request_id="request-0",
        new_token_ids=[],
        ec_transfer_params=transfer,
    )

    assert output.ec_transfer_params == transfer
