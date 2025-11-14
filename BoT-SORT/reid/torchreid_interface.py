import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from torchreid import models
from torchreid.utils import load_pretrained_weights


_RESIZE = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
])


class TorchreIDInterface:
    """
    Lightweight wrapper that loads a torchreid model and exposes an inference
    method compatible với BoTSORT.
    """

    def __init__(self, weight_path: str, device: str = "cuda", batch_size: int = 16):
        if device != "cpu" and torch.cuda.is_available():
            self.device = torch.device(device)
        else:
            self.device = torch.device("cpu")

        self.model = models.build_model("osnet_x1_0", num_classes=0, pretrained=False)
        load_pretrained_weights(self.model, weight_path)
        self.model.to(self.device).eval()

        self.batch_size = max(1, int(batch_size))

    @torch.no_grad()
    def inference(self, image, dets_xyxy):
        if dets_xyxy is None or len(dets_xyxy) == 0:
            return []

        h, w = image.shape[:2]
        patches = []
        for tlbr in dets_xyxy:
            x1, y1, x2, y2 = tlbr.astype(int)
            x1 = max(0, min(w - 1, x1))
            x2 = max(0, min(w - 1, x2))
            y1 = max(0, min(h - 1, y1))
            y2 = max(0, min(h - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue

            crop = image[y1:y2, x1:x2, ::-1]  # BGR -> RGB
            tensor = _RESIZE(crop).to(self.device)
            patches.append(tensor)

        if not patches:
            return []

        features = []
        for idx in range(0, len(patches), self.batch_size):
            batch = torch.stack(patches[idx: idx + self.batch_size])
            emb = self.model(batch)
            emb = F.normalize(emb, dim=1)
            features.append(emb.cpu().numpy())

        return np.concatenate(features, axis=0) if features else []
