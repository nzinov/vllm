# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.engine.arg_utils import EngineArgs
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.exceptions import SchedulerWaitingQueueFullError
from vllm.v1.engine.llm_engine import LLMEngine as V1LLMEngine


def test_waiting_queue_full(monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")
        m.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

        engine_args = EngineArgs(
            model="facebook/opt-125m",
            enforce_eager=True,
            max_waiting_queue_length=1,
        )
        engine = V1LLMEngine.from_engine_args(engine_args=engine_args)

        sampling_params = SamplingParams(max_tokens=1)
        engine.add_request("0", "foo", sampling_params)

        with pytest.raises(SchedulerWaitingQueueFullError):
            engine.add_request("1", "bar", sampling_params)

        engine.shutdown()
