#!/usr/bin/env python3
"""Export a trained PETR Wi-Fi checkpoint to ONNX (input: one CSI window; outputs: per-slot score, keypoints).

    python csi_demo/infer/export_onnx.py CONFIG CKPT SAMPLE_NPZ OUT_ONNX

Runs on CPU without MMCV's compiled ops (they are only needed by image models; this Wi-Fi PETR uses plain
MultiheadAttention), so a Python 3.8 env with torch 1.13.1 + mmcv 1.5.3 built without ops is enough.
Checks the exported graph against PyTorch on 20 windows of SAMPLE_NPZ (key 'csi').
"""
import importlib.abc
import importlib.machinery
import sys
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / 'third_party/mmdet')]


class _Dummy:
    def __init__(self, *a, **k):
        pass


class _StubModule(types.ModuleType):
    __path__ = []

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        cls = type(name, (_Dummy,), {})
        setattr(self, name, cls)
        return cls


class _StubOps(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path, target=None):
        if name == 'mmcv.ops' or name.startswith('mmcv.ops.') or name == 'mmcv._ext':
            return importlib.machinery.ModuleSpec(name, self, is_package=True)

    def create_module(self, spec):
        return _StubModule(spec.name)

    def exec_module(self, module):
        pass


sys.meta_path.insert(0, _StubOps())

import torch  # noqa: E402
from mmcv import Config  # noqa: E402
from mmcv.runner import load_checkpoint  # noqa: E402

from opera.models import build_model  # noqa: E402


class Wrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, csi):
        feat = self.m.head(csi.reshape(1, -1, csi.shape[-1]))
        bboxes, labels, kpts = self.m.bbox_head.simple_test(feat, [dict()])[0]
        return bboxes[:, 4], kpts[..., :2]


def main():
    cfg_path, ckpt, sample, out = sys.argv[1:5]
    model = build_model(Config.fromfile(cfg_path).model)
    load_checkpoint(model, ckpt, map_location='cpu')
    w = Wrap(model.eval()).eval()
    csi = np.load(sample)['csi'].astype(np.float32)
    torch.onnx.export(w, torch.from_numpy(csi[:1]), out, opset_version=17, input_names=['csi'],
                      output_names=['score', 'kpts'], do_constant_folding=True)
    import onnxruntime as ort
    sess = ort.InferenceSession(out, providers=['CPUExecutionProvider'])
    same, top1 = 0, 0.0
    for i in range(20):
        x = csi[i:i + 1]
        with torch.no_grad():
            s, k = (t.numpy() for t in w(torch.from_numpy(x)))
        os_, ok = sess.run(None, {'csi': x})
        same += int(np.argmax(os_) == np.argmax(s))
        top1 = max(top1, float(np.abs(ok[np.argmax(os_)] - k[np.argmax(s)]).max()))
    print(f'exported {out}: input {csi[:1].shape}; same top-1 slot {same}/20, top-1 keypoint max diff {top1:.2e}')


if __name__ == '__main__':
    main()
