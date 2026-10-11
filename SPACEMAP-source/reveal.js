// Fixed demonstration data, evaluated from one playhead. Time pauses at each
// event boundary until the next router is clicked. Radius is a visual sweep.
export function createRevealTimeline(settings, layout) {
  const events = settings.timelineSeconds.filter(item => item.action === 'reveal').map(item => {
    const region = layout.regions.find(region => region.id === item.region);
    const router = layout.routers.find(router => router.id === item.router);
    const person = layout.people.find(person => person.id === item.person);
    if (!region || !router || !person || !(item.start < item.signalStart && item.signalStart < item.end) || !item.entryPath || item.entryPath.length < 2) {
      throw new Error(`공개 설정을 확인해 주세요: ${item.region}`);
    }
    const radius = Math.max(...region.x.flatMap(x => region.z.map(z =>
      Math.hypot(x - router.position[0], z - router.position[2]))));
    const lengths = item.entryPath.slice(1).map((point, i) => Math.hypot(point[0] - item.entryPath[i][0], point[1] - item.entryPath[i][1]));
    return { ...item, bounds: region, origin: router.position, radius, lengths, pathLength: lengths.reduce((a, b) => a + b, 0) };
  });
  const start = events[0].start;
  const end = events.at(-1).end;
  function entryAt(time) {
    const trail = [events[0].entryPath[0]];
    let direction = [0, 1];
    for (const event of events) {
      let remaining = Math.max(0, Math.min(1, (time - event.start) / (event.signalStart - event.start))) * event.pathLength;
      for (let i = 0; i < event.lengths.length; i++) {
        const a = event.entryPath[i], b = event.entryPath[i + 1], length = event.lengths[i];
        if (remaining <= 0) break;
        direction = [(b[0] - a[0]) / length, (b[1] - a[1]) / length];
        const fraction = Math.min(1, remaining / length);
        trail.push([a[0] + (b[0] - a[0]) * fraction, a[1] + (b[1] - a[1]) * fraction]);
        remaining -= length;
      }
      if (time < event.signalStart) break;
    }
    return { position: trail.at(-1), direction, trail };
  }
  return {
    events, start, end, holdAt: end,
    at(time, confirmedPeople = new Set()) {
      const regions = events.map(event => {
        const fraction = Math.max(0, Math.min(1, (time - event.signalStart) / (event.end - event.signalStart)));
        return { ...event, progress: fraction * fraction * (3 - 2 * fraction), complete: time >= event.end };
      });
      return {
        time,
        regions,
        // Geometry is revealed by time; people appear only after confirmation.
        visiblePeople: new Set(regions.filter(region => region.complete && confirmedPeople.has(region.person)).map(region => region.person)),
        started: time >= start,
        complete: time >= end,
        active: regions.find(region => time >= region.start && time < region.end),
        entry: entryAt(time),
      };
    },
  };
}
