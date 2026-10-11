// Local synthetic joint coordinates. The caller supplies the selected person ID
// separately, so this sample can later be replaced by another coordinate source.
export function samplePose(sample, elapsedMs, output) {
  const frames = sample.frames;
  const time = Math.max(0, Math.min(sample.durationMs, elapsedMs));
  let next = 1;
  while (next < frames.length - 1 && frames[next].timeMs < time) next++;
  const a = frames[next - 1], b = frames[next];
  const fraction = (time - a.timeMs) / (b.timeMs - a.timeMs);
  for (const [id, start] of Object.entries(a.joints)) {
    const end = b.joints[id];
    for (let axis = 0; axis < 3; axis++) output[id][axis] = start[axis] + (end[axis] - start[axis]) * fraction;
  }
  return output;
}
