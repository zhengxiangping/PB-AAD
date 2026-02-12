import os

from easytorch.utils.registry import scan_modules

from .registry import SCALER_REGISTRY
from .dataset import TimeSeriesForecastingDataset

# 在 Jupyter Notebook 环境中,直接导入 transform 模块以确保装饰器被执行
try:
    from . import transform
except ImportError:
    pass

__all__ = ["SCALER_REGISTRY", "TimeSeriesForecastingDataset"]

# fix bugs on Windows systems and on jupyter
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    scan_modules(project_dir, __file__, ["__init__.py", "registry.py"])
except (TypeError, ImportError) as e:
    # 在 Jupyter Notebook 环境中,scan_modules 可能失败
    # 但我们已经通过直接导入 transform 模块来注册 scaler 函数
    pass
# scan_modules(project_dir, __file__, ["basicts.data.dataset", "basicts.data.registry"])
# scan_modules(project_dir, __file__, ["dataset", "registry"])
