"""v2 第 2 步：图像 backbone —— 预训练 ResNet-18 逐相机提特征。

把 6 路环视相机图像各自过同一个（共享权重）ResNet-18，输出每路的特征图。
第一版 **冻结 CNN**（当作 ImageNet 预训练的通用视觉特征提取器），后续 BEV +
规划头跑通后再联合微调。

记号: B=batch, V=相机数(6), C_in=3, H/W=输入图高宽(224/400),
      C_feat=特征通道, Hf/Wf=特征图高宽（ResNet 下采样 32 倍）。
"""

import torch
import torch.nn as nn
from torch import Tensor


class ImageBackbone(nn.Module):
    """共享权重的 ResNet-18 特征提取器，批量处理 V 路相机图像。

    - 去掉 avgpool + fc，保留到最后一个卷积 stage，输出空间特征图（下采样 32x）。
    - 6 路图合并到 batch 维一次性过 CNN（共享权重），再 reshape 回 (B, V, ...)。
    - freeze=True 时参数 requires_grad=False 且 eval()（冻结通用视觉特征）。
    """

    out_channels = 512  # ResNet-18 最后 stage 的通道数

    def __init__(self, pretrained: bool = True, freeze: bool = True):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = resnet18(weights=weights)
        # 保留 conv1..layer4（去掉 avgpool/fc），得到空间特征图
        self.stem = nn.Sequential(
            net.conv1, net.bn1, net.relu, net.maxpool,
            net.layer1, net.layer2, net.layer3, net.layer4)
        # ImageNet 归一化常数（输入须为 [0,1] 的 RGB，再做标准化）
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self._frozen = freeze
        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    def train(self, mode: bool = True):
        """冻结时始终保持 eval（固定 BN 统计），即使外部调用 .train()。"""
        super().train(mode)
        if self._frozen:
            self.stem.eval()
        return self

    def forward(self, images: Tensor) -> Tensor:
        """
        输入:
          images (B, V, 3, H, W)  —— V 路相机图像，像素已归一到 [0,1] 的 RGB
        输出:
          feats  (B, V, C_feat, Hf, Wf)  —— 每路相机的特征图（C_feat=512）
        """
        B, V, C, H, W = images.shape
        x = images.reshape(B * V, C, H, W)
        x = (x - self.mean) / self.std            # ImageNet 标准化
        feat = self.stem(x)                        # (B*V, 512, Hf, Wf)
        Cf, Hf, Wf = feat.shape[1:]
        return feat.reshape(B, V, Cf, Hf, Wf)


if __name__ == "__main__":
    # 形状 / 冻结 / 显存 自测：用随机输入（B=1, V=6, 3, 224, 400）跑一次前向。
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # pretrained=False 以免首次运行联网下载；仅验证结构与形状。
    bb = ImageBackbone(pretrained=False, freeze=True).to(dev)
    bb.eval()

    B, V, H, W = 1, 6, 224, 400
    imgs = torch.rand(B, V, 3, H, W, device=dev)  # [0,1]
    with torch.no_grad():
        feats = bb(imgs)

    print(f"输入  images: {tuple(imgs.shape)}")
    print(f"输出  feats : {tuple(feats.shape)}  (期望 (1,6,512,7,13))")

    n_total = sum(p.numel() for p in bb.parameters())
    n_train = sum(p.numel() for p in bb.parameters() if p.requires_grad)
    print(f"参数量: 总计 {n_total:,}  可训练 {n_train:,}  (冻结则可训练=0)")

    if dev == "cuda":
        torch.cuda.synchronize()
        mem = torch.cuda.max_memory_allocated() / 1024**2
        print(f"峰值显存(单样本前向): {mem:.1f} MiB")
