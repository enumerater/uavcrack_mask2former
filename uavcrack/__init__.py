"""UAV-Crack 比赛专用组件 (mmseg 插件)。

放在这里而不是直接改 mmseg 源码, 好处是 mmseg 那边升级/更新代码不会覆盖掉。
通过 tools/train.py 等入口脚本把本项目的根目录加进 sys.path 后,
在 config 里用 custom_imports 导入即可触发注册。
"""
from .metrics import UAVCrackMetric  # noqa: F401

__all__ = ['UAVCrackMetric']
