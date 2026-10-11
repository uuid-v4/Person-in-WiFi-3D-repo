import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

// Hosting transport only: reconstruct the exact original GLB before parsing.
export async function loadModel(modelUrl, onProgress) {
  const url = new URL(modelUrl, location.href);
  const response = await fetch(`${url}.json`);
  if (!response.ok) throw new Error(`모델 정보: HTTP ${response.status}`);
  const manifest = await response.json();
  let loaded = 0;
  const parts = await Promise.all(manifest.parts.map(async name => {
    const part = await fetch(new URL(name, url));
    if (!part.ok) throw new Error(`모델 파일: HTTP ${part.status}`);
    const buffer = await part.arrayBuffer();
    loaded += buffer.byteLength;
    onProgress?.({ lengthComputable: true, loaded, total: manifest.bytes });
    return buffer;
  }));
  const stream = new Blob(parts).stream().pipeThrough(new DecompressionStream('gzip'));
  const bytes = await new Response(stream).arrayBuffer();
  return new GLTFLoader().parseAsync(bytes, new URL('.', url).href);
}
