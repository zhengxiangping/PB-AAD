import functools
import torch
def select_input_features(data: torch.Tensor,forward_features) -> torch.Tensor:
    """Select input features.

    Args:
        data (torch.Tensor): input history data, shape [B, L, N, C]

    Returns:
        torch.Tensor: reshaped data
    """

    # select feature using self.forward_features
    if forward_features is not None:
        data = data[:, :, :, forward_features]
    return data

def select_target_features(data: torch.Tensor,target_features) -> torch.Tensor:
    """Select target feature.

    Args:
        data (torch.Tensor): prediction of the model with arbitrary shape.

    Returns:
        torch.Tensor: reshaped data with shape [B, L, N, C]
    """

    # select feature using self.target_features
    data = data[:, :, :, target_features]
    return data

def metric_forward(metric_func, args):
    """Computing metrics.

    Args:
        metric_func (function, functools.partial): metric function.
        args (list): arguments for metrics computation.
    """

    if isinstance(metric_func, functools.partial) and list(metric_func.keywords.keys()) == ["null_val"]:
        # support partial(metric_func, null_val = something)
        metric_item = metric_func(*args)
    elif callable(metric_func):
        # is a function
        metric_item = metric_func(*args, null_val=0.0)
    else:
        raise TypeError("Unknown metric type: {0}".format(type(metric_func)))
    return metric_item
