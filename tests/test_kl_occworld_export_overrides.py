from types import SimpleNamespace

import numpy as np
import torch

from tools.analysis_tools.export_kl_occworld_predictions import (
    _override_current_anchor,
)


def test_exporter_overrides_current_anchor_without_touching_history():
    state = np.full((10, 120, 160), 3, dtype=np.int64)
    valid = np.ones((10, 120, 160), dtype=np.bool_)
    current_state = torch.zeros((1, 10, 120, 160), dtype=torch.long)
    current_valid = torch.zeros((1, 10, 120, 160), dtype=torch.bool)
    history = torch.full((1, 5, 10, 120, 160), 7, dtype=torch.long)
    batch = {
        'current_world_state': SimpleNamespace(data=[current_state]),
        'current_world_valid': SimpleNamespace(data=[current_valid]),
        'history_world_state': SimpleNamespace(data=[history]),
    }

    _override_current_anchor(batch, state, valid)

    assert torch.equal(current_state[0], torch.from_numpy(state))
    assert torch.equal(current_valid[0], torch.from_numpy(valid))
    assert torch.all(history == 7)
