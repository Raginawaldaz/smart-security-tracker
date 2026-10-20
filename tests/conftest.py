# 模块：pytest 共享 fixture
# 作用：提供临时数据目录，避免测试写入仓库里的 data/。

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    path = tmp_path / "data"
    path.mkdir()
    return path
