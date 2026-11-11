from collections import OrderedDict

from torch import nn
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present


def load_model_weights(model, state_dict):
    target = model.module if isinstance(model, nn.DataParallel) else model
    weights = OrderedDict(state_dict)
    if hasattr(state_dict, '_metadata'):
        weights._metadata = state_dict._metadata.copy()
    consume_prefix_in_state_dict_if_present(weights, 'module.')
    target.load_state_dict(weights)
