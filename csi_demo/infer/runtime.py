"""Run the exported PETR ONNX model on one CSI window with ClikaRT (default) or onnxruntime.

    rt = PoseRuntime('data/ckpt/petr_v2.onnx', backend='clikart')
    score, kpts = rt.predict(window)   # window (rx, tx, 20, F) float32 -> top-1 score, (14, 2) in [0, 1]

ClikaRT needs its license: set CLIKA_RT_LICENSE to the credential file path before the first call
(PoseRuntime does this from the `license_path` argument if given).
"""
import os

import numpy as np


class PoseRuntime:
    def __init__(self, onnx_path, backend='clikart', license_path=None):
        self.backend = backend
        if backend == 'clikart':
            if license_path:
                os.environ['CLIKA_RT_LICENSE'] = str(license_path)
            import clika_runtime as crt
            self._crt = crt
            graph = crt.io.OnnxModel.open(str(onnx_path)).compile()
            graph = graph.value() if hasattr(graph, 'value') else graph
            graph.optimize()
            graph.finalize()
            self._graph = graph
        elif backend == 'onnxruntime':
            import onnxruntime as ort
            self._sess = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
        else:
            raise ValueError(f'unknown backend {backend!r}')

    def raw(self, window):
        x = np.ascontiguousarray(window[None], dtype=np.float32)
        if self.backend == 'onnxruntime':
            score, kpts = self._sess.run(None, {'csi': x})
            return np.asarray(score).reshape(-1), np.asarray(kpts).reshape(-1, 14, 2)
        outs = [np.array(t.numpy() if hasattr(t, 'numpy') else t) for t in self._graph.run([self._crt.as_tensor(x)])]
        score = next(a for a in outs if a.ndim == 1 or a.shape[-1] == 1).reshape(-1)
        kpts = next(a for a in outs if a.shape[-2:] == (14, 2)).reshape(-1, 14, 2)
        return score, kpts

    def predict(self, window):
        score, kpts = self.raw(window)
        i = int(np.argmax(score))
        return float(score[i]), kpts[i]
