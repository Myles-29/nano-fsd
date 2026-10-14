"""任务 3.7 单元测试：freeze_perception_stack 的冻结正确性（属性 3 静态部分）。

验证: 正确性属性 3（静态部分）
调用 freeze_perception_stack 后：
  - backbone / perception_head 全部参数 requires_grad == False
  - planning_head 全部参数 requires_grad == True
  - backbone / perception_head 置为 eval() 模式
  - model.parameters() 中 requires_grad==True 的参数恰好是 planning_head 的参数
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config  # noqa: E402
from model import NanoFSD  # noqa: E402


def build_model() -> NanoFSD:
    """构建一个 NanoFSD 并调用 freeze_perception_stack。"""
    model = NanoFSD(Config())
    model.freeze_perception_stack()
    return model


def test_perception_stack_params_frozen():
    """backbone 与 perception_head 的所有参数 requires_grad==False。"""
    model = build_model()
    for module in (model.backbone, model.perception_head):
        params = list(module.parameters())
        assert len(params) > 0  # 确保确实遍历到参数
        for p in params:
            assert p.requires_grad is False


def test_planning_head_params_trainable():
    """planning_head 的所有参数 requires_grad==True。"""
    model = build_model()
    params = list(model.planning_head.parameters())
    assert len(params) > 0
    for p in params:
        assert p.requires_grad is True


def test_perception_stack_in_eval_mode():
    """backbone 与 perception_head 置为 eval()，planning_head 参数仍可训练。"""
    model = build_model()
    assert model.backbone.training is False
    assert model.perception_head.training is False
    # 规划头仍为可训练参数
    assert all(p.requires_grad for p in model.planning_head.parameters())


def test_trainable_params_are_exactly_planning_head():
    """model.parameters() 中可训练参数恰好等于 planning_head 的参数集合。"""
    model = build_model()
    trainable = [p for p in model.parameters() if p.requires_grad]
    planning = list(model.planning_head.parameters())

    assert len(trainable) == len(planning)
    # 用对象 id 做身份比对，确保是同一批张量
    trainable_ids = {id(p) for p in trainable}
    planning_ids = {id(p) for p in planning}
    assert trainable_ids == planning_ids
