import json
import logging
import random

import numpy as np
import pytest
import torch

from chestxray.utils import (
    collect_environment_info,
    get_device,
    save_environment_info,
    set_seed,
    setup_logging,
)


def draw():
    return random.random(), np.random.rand(), torch.rand(3)


def test_set_seed_is_reproducible():
    set_seed(123)
    first = draw()
    set_seed(123)
    second = draw()
    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])
    assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark


def test_get_device():
    assert get_device("cpu") == torch.device("cpu")
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert get_device("auto").type == expected


@pytest.mark.skipif(torch.cuda.is_available(), reason="only meaningful without CUDA")
def test_get_device_cuda_unavailable_raises():
    with pytest.raises(RuntimeError):
        get_device("cuda")


def test_environment_info(tmp_path):
    info = save_environment_info(tmp_path / "env.json")
    assert json.loads((tmp_path / "env.json").read_text(encoding="utf-8")) == info
    assert info["torch"] == torch.__version__
    for key in ("python", "cuda", "cudnn", "gpus", "git_commit", "packages"):
        assert key in info
    assert info["packages"]["numpy"] == np.__version__
    assert len(info["gpus"]) == torch.cuda.device_count()


def test_setup_logging_writes_file(tmp_path):
    log_file = tmp_path / "logs" / "train.log"
    setup_logging(log_file)
    logging.getLogger("chestxray.some_module").info("hello from a submodule")
    logging.getLogger("__main__").info("hello from a script")
    for handler in logging.getLogger().handlers:
        handler.flush()
    text = log_file.read_text(encoding="utf-8")
    assert "hello from a submodule" in text and "hello from a script" in text

    setup_logging()  # closes the file handler so tmp_path can be removed on Windows
    ours = [h for h in logging.getLogger().handlers if getattr(h, "_chestxray", False)]
    assert len(ours) == 1 and not isinstance(ours[0], logging.FileHandler)
