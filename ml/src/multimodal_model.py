"""Late-fusion multimodal classifier: YOLOv8 image backbone + tabular MLP.

Architecture::

    Image:   [B,3,224,224] → YOLO backbone+conv → pool → [B,1280] → Linear → [B,512]
    Tabular: [B,102]       → MLP(102→256→128)                          → [B,128]
    Fusion:  concat [B,640] → MLP → [B, num_classes]
"""

import logging

import torch
import torch.nn as nn

from src.feature_engineering import TABULAR_DIM

logger = logging.getLogger(__name__)

_IMAGE_FEAT_DIM = 1280  # YOLOv8n-cls feature dimension after Classify conv+pool
_IMAGE_PROJ_DIM = 512
_TAB_HIDDEN = 256
_TAB_OUT = 128
_FUSED_HIDDEN = 256


class MultimodalGameClassifier(nn.Module):
    """YOLO image backbone + tabular MLP with late fusion.

    Parameters
    ----------
    tabular_dim:
        Length of the tabular feature vector (default ``TABULAR_DIM``).
    num_classes:
        Number of output classes (default 2: Gaming / Not Gaming).
    yolo_model:
        Ultralytics model identifier or path to ``.pt`` weights.
    """

    def __init__(
        self,
        tabular_dim: int = TABULAR_DIM,
        num_classes: int = 2,
        yolo_model: str = "yolov8n-cls.pt",
    ) -> None:
        super().__init__()

        # ---- Image branch: YOLO backbone ----
        self._build_image_branch(yolo_model)

        # ---- Image projection ----
        self.image_proj = nn.Sequential(
            nn.Linear(_IMAGE_FEAT_DIM, _IMAGE_PROJ_DIM),
            nn.ReLU(),
        )

        # ---- Tabular branch ----
        self.tabular_mlp = nn.Sequential(
            nn.Linear(tabular_dim, _TAB_HIDDEN),
            nn.BatchNorm1d(_TAB_HIDDEN),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(_TAB_HIDDEN, _TAB_OUT),
            nn.BatchNorm1d(_TAB_OUT),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # ---- Fusion classifier ----
        self.classifier = nn.Sequential(
            nn.Linear(_IMAGE_PROJ_DIM + _TAB_OUT, _FUSED_HIDDEN),
            nn.BatchNorm1d(_FUSED_HIDDEN),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(_FUSED_HIDDEN, num_classes),
        )

        logger.info(
            "MultimodalGameClassifier: tabular_dim=%d classes=%d yolo=%s",
            tabular_dim, num_classes, yolo_model,
        )

    # ------------------------------------------------------------------
    # Image branch construction
    # ------------------------------------------------------------------

    def _build_image_branch(self, yolo_model: str) -> None:
        """Extract the YOLO backbone + Classify conv/pool as plain nn.Modules."""
        from ultralytics import YOLO

        yolo = YOLO(yolo_model)
        children = list(yolo.model.model.children())
        classify_head = children[-1]

        # Backbone: Conv/C2f layers (everything except Classify)
        self.backbone = nn.Sequential(*children[:-1])

        # Classify head's 1×1 conv (256→1280) + BN + SiLU
        self.backbone_conv = classify_head.conv

        # Global average pool
        self.backbone_pool = nn.AdaptiveAvgPool2d(1)
        self.backbone_flatten = nn.Flatten()

        # Verify expected feature dimension
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            feat = self._image_features(dummy)
        actual_dim = feat.shape[1]
        if actual_dim != _IMAGE_FEAT_DIM:
            raise RuntimeError(
                f"Expected YOLO backbone to produce {_IMAGE_FEAT_DIM}-d features, "
                f"got {actual_dim}. Check ultralytics version."
            )

    def _image_features(self, x: torch.Tensor) -> torch.Tensor:
        """Run the image through backbone → conv → pool → flatten."""
        x = self.backbone(x)
        x = self.backbone_conv(x)
        x = self.backbone_pool(x)
        return self.backbone_flatten(x)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, image: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        image:  [B, 3, 224, 224]
        tabular: [B, tabular_dim]

        Returns
        -------
        logits: [B, num_classes]
        """
        x_img = self._image_features(image)     # [B, 1280]
        x_img = self.image_proj(x_img)           # [B, 512]

        x_tab = self.tabular_mlp(tabular)        # [B, 128]

        x = torch.cat([x_img, x_tab], dim=1)    # [B, 640]
        return self.classifier(x)                # [B, num_classes]

    # ------------------------------------------------------------------
    # Freeze / unfreeze helpers
    # ------------------------------------------------------------------

    def freeze_backbone(self) -> None:
        """Freeze YOLO backbone + conv weights (no gradient updates)."""
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.backbone_conv.parameters():
            p.requires_grad = False
        logger.info("Backbone frozen")

    def unfreeze_backbone(self) -> None:
        """Unfreeze YOLO backbone + conv weights for fine-tuning."""
        for p in self.backbone.parameters():
            p.requires_grad = True
        for p in self.backbone_conv.parameters():
            p.requires_grad = True
        logger.info("Backbone unfrozen")

    def get_backbone_params(self) -> list[nn.Parameter]:
        """Return backbone + conv parameters (for differential LR)."""
        return list(self.backbone.parameters()) + list(self.backbone_conv.parameters())

    def get_head_params(self) -> list[nn.Parameter]:
        """Return all non-backbone parameters."""
        backbone_ids = {id(p) for p in self.get_backbone_params()}
        return [p for p in self.parameters() if id(p) not in backbone_ids]
