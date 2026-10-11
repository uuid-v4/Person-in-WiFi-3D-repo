// Live Wi-Fi pose from the CSI hub (csi_demo/live/hub.py, GET /live/pose).
// The model emits 14 normalized 2D image keypoints in the paper's slot order; this module lifts them
// to the dashboard's person-local metres (y up, origin between the feet) with a fixed depth.
const SLOT_IDS = ['neck', 'head', 'left_shoulder', 'right_shoulder', 'left_elbow', 'left_hip', 'right_elbow',
  'right_hip', 'left_wrist', 'left_knee', 'right_wrist', 'right_knee', 'left_ankle', 'right_ankle'];
const IMAGE_ASPECT = 4 / 3;
const MAX_AGE_S = 2.5;
const POLL_MS = 200;
const SMOOTH_TAU_MS = 140;
const MIN_SPAN = 0.05;

export function createLivePose(hubUrl, skeleton) {
  const enabled = Boolean(hubUrl) && hubUrl !== 'off';
  const neutral = Object.fromEntries(skeleton.joints.map(joint => [joint.id, joint.position]));
  const standHeight = neutral.head[1] - neutral.left_ankle[1];
  const floorY = neutral.left_ankle[1];
  let timer = 0, latest = null, target = null, current = null, scale = null;

  async function poll() {
    const controller = new AbortController();
    const abort = setTimeout(() => controller.abort(), 1000);
    try {
      const response = await fetch(`${hubUrl}/live/pose`, { cache: 'no-store', signal: controller.signal });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      latest = { payload: await response.json(), at: performance.now() };
      target = lift(latest.payload.kpts);
    } catch {
      latest = null;
      target = null;
    } finally {
      clearTimeout(abort);
    }
  }

  function lift(kpts) {
    if (!Array.isArray(kpts) || kpts.length !== SLOT_IDS.length) return null;
    const p = Object.fromEntries(SLOT_IDS.map((id, slot) => [id, kpts[slot]]));
    const hipX = (p.left_hip[0] + p.right_hip[0]) / 2;
    const ankleY = (p.left_ankle[1] + p.right_ankle[1]) / 2;
    const span = ankleY - p.head[1];
    if (!(span > MIN_SPAN)) return null;
    const fit = Math.min(Math.max(standHeight / span, standHeight / 1), standHeight / 0.25);
    scale = scale === null ? fit : 0.9 * scale + 0.1 * fit;
    const joints = {};
    for (const id of SLOT_IDS) {
      joints[id] = [-(p[id][0] - hipX) * IMAGE_ASPECT * scale, floorY + (ankleY - p[id][1]) * scale, neutral[id][2]];
    }
    return joints;
  }

  function fresh() {
    if (!latest || !target || !latest.payload.kpts) return false;
    const age = (latest.payload.age ?? 0) + (performance.now() - latest.at) / 1000;
    return age <= MAX_AGE_S;
  }

  return {
    start() {
      if (!enabled || timer) return;
      poll();
      timer = setInterval(poll, POLL_MS);
    },
    active: fresh,
    step(deltaMs, output) {
      if (!fresh()) { current = null; return null; }
      const alpha = current ? 1 - Math.exp(-deltaMs / SMOOTH_TAU_MS) : 1;
      current ??= Object.fromEntries(SLOT_IDS.map(id => [id, [...target[id]]]));
      for (const id of SLOT_IDS) {
        for (let axis = 0; axis < 3; axis++) current[id][axis] += (target[id][axis] - current[id][axis]) * alpha;
        output[id][0] = current[id][0];
        output[id][1] = current[id][1];
        output[id][2] = current[id][2];
      }
      return output;
    },
    summary() {
      const payload = latest?.payload;
      if (!payload) return '허브 연결 대기';
      const rate = payload.hz ? `${payload.hz.toFixed(1)} Hz` : '-- Hz';
      const delay = payload.latency_ms ? ` · ${Math.round(payload.latency_ms)} ms` : '';
      return `${rate}${delay}`;
    },
  };
}
