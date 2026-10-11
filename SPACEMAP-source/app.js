import { createMap } from './map-view.js?v=4';
import { createBuilding } from './building-view.js?v=8';
import { createRevealTimeline } from './reveal.js?v=7';
import { samplePose } from './pose-sample.js?v=5';
import { createLivePose } from './live-pose.js?v=1';
import { observationSample } from './observation-sample.js?v=6';

const title = document.querySelector('#status-title');
const detail = document.querySelector('#status-detail');
const mapLayer = document.querySelector('#map-layer');
const buildingLayer = document.querySelector('#building-layer');
const progressFill = document.querySelector('#progress-fill');
const peopleCount = document.querySelector('#people-count');
const routeTarget = document.querySelector('#route-target');
const pauseButton = document.querySelector('#pause-button');
const playbackControls = document.querySelector('#playback-controls');
const posePanel = document.querySelector('#pose-panel');
const poseStatus = document.querySelector('#pose-status');
const poseEyebrow = document.querySelector('#pose-eyebrow');
const poseNote = document.querySelector('#pose-note');
const overviewButton = document.querySelector('#overview-button');
const observationControls = document.querySelector('#observation-controls');
const observationButton = document.querySelector('#observation-button');
const observationPanel = document.querySelector('#observation-panel');
const confirmationButton = document.querySelector('#confirmation-button');
const recordButton = document.querySelector('#record-button');
const resultsButton = document.querySelector('#results-button');
const recordMemo = document.querySelector('#record-memo');
const resultsDetails = document.querySelector('#results-details');
const viewButtons = [...document.querySelectorAll('[data-view]')];
const smooth = value => { const t = Math.max(0, Math.min(1, value)); return t * t * (3 - 2 * t); };
const progress = (time, interval) => smooth((time - interval.start) / (interval.end - interval.start));

// Simulation time is independent of the user's camera and stops between clicks.
export const playback = { time: 0, selected: false, running: false, paused: false, ready: false, phase: 'loading' };

function status(phase, heading, description, step = 'map') {
  if (playback.phase === phase) return;
  playback.phase = phase;
  document.body.dataset.phase = phase;
  title.textContent = heading;
  detail.textContent = description;
  document.querySelectorAll('[data-step]').forEach(item => {
    if (item.dataset.step === step) item.setAttribute('aria-current', 'step');
    else item.removeAttribute('aria-current');
  });
}

async function init() {
  async function readData(path) {
    const response = await fetch(path);
    if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
    return response.json();
  }
  const settings = await readData('./preparation/demo-settings.json');
  const [layout, pose, poseSample] = await Promise.all([
    readData(settings.layoutFile), readData(settings.skeletonDefinition),
    readData('./preparation/mock-pose-5s.json'),
  ]);
  const timeline = createRevealTimeline(settings, layout);
  const focus = settings.timelineSeconds.find(item => item.action === 'map-focus');
  const exterior = settings.timelineSeconds.find(item => item.action === 'building-exterior-only');
  const top = settings.timelineSeconds.find(item => item.action === 'top-view-interior-masked');
  if (![focus, exterior, top].every(item => item && item.end > item.start)) throw new Error('화면 전환 시간 설정을 확인해 주세요.');
  document.querySelector('#address').textContent = settings.map.roadAddress || settings.map.venueAddress;
  const map = createMap(settings.map, start, () => {
    const notice = document.querySelector('#map-notice');
    notice.hidden = false;
    notice.textContent = '지도 타일을 불러오지 못했습니다. 인터넷 연결을 확인해 주세요.';
  });
  let building, segment, animationFrame = 0;
  let selection = null;
  let entryStarted = false;
  const confirmedPeople = new Set();
  const records = new Map();
  let detailView = null, recordId = null;
  const observation = { phase: 'idle', elapsed: 0, time: 0, lastAt: null, selectedId: null };
  const observations = new Map(observationSample.points.map(point => [point.person, point]));
  function currentEvent() {
    return timeline.events.find(event => !confirmedPeople.has(event.person));
  }
  function awaitingConfirmation() {
    const current = currentEvent();
    return entryStarted && current && playback.time >= current.end ? current : null;
  }
  function hasObservationMotion() {
    return observation.phase === 'running' || (observation.phase === 'complete' && confirmedPeople.size < layout.people.length);
  }
  const sampledJoints = Object.fromEntries(pose.joints.map(joint => [joint.id, [0, 0, 0]]));
  const hubParam = new URLSearchParams(location.search).get('hub');
  const live = createLivePose(hubParam === null ? 'http://127.0.0.1:8766' : hubParam, pose);
  function scheduleFrame() {
    if (!animationFrame && !playback.paused && (playback.running || selection?.phase === 'playing' || (playback.selected && playback.time >= timeline.start))) {
      animationFrame = requestAnimationFrame(frame);
    }
  }
  function clearSelection() {
    selection = null;
    building.clearPersonSelection();
    building.resetPose();
    // A sample-only pause should not leave the router controls locked.
    if (!segment && !hasObservationMotion()) playback.paused = false;
  }
  function selectPerson(personId) {
    if (!timeline.at(playback.time, confirmedPeople).visiblePeople.has(personId)) return;
    live.start();
    observation.selectedId = null;
    detailView = null;
    recordId = null;
    if (selection) building.resetPose(selection.personId);
    const current = { personId, phase: 'focusing', elapsed: 0, lastAt: 0 };
    selection = current;
    const accepted = building.focusPerson(personId, () => {
      if (selection !== current) return;
      current.phase = 'playing';
      current.lastAt = performance.now();
      render();
      scheduleFrame();
    });
    if (!accepted) selection = null;
    render();
  }
  function runTo(end) {
    segment = { start: playback.time, end, at: performance.now() };
    playback.running = true;
    playback.paused = false;
    render();
    scheduleFrame();
  }
  function start() {
    if (!playback.ready || playback.selected) return;
    playback.selected = true;
    map.lock();
    status('map-focus', '선택한 장소로 이동합니다', '주소 참조점을 확대하고 있습니다.');
    runTo(timeline.start);
  }
  function clickRouter(routerId) {
    if (!entryStarted || !playback.ready || !playback.selected || playback.running || playback.paused || playback.time < timeline.start) return;
    const next = currentEvent();
    if (!next || next.router !== routerId || playback.time >= next.end) return;
    runTo(next.end);
  }
  function selectObservation(personId) {
    if (observation.phase !== 'complete' || timeline.at(playback.time, confirmedPeople).visiblePeople.has(personId)) return;
    const point = observations.get(personId);
    const person = layout.people.find(person => person.id === personId);
    if (!point || !person) return;
    live.start();
    detailView = null;
    recordId = null;
    observation.selectedId = personId;
    document.querySelector('#observation-region').textContent = `${person.region}구역`;
    document.querySelector('#observation-movement').textContent = point.movement ? '있음' : '없음';
    document.querySelector('#observation-person').textContent = point.personPossible ? '가능성 있음' : '판단 보류';
    document.querySelector('#observation-level').textContent = point.movementLevel;
    render();
  }
  observationButton.addEventListener('click', () => {
    if (!playback.selected || playback.time < timeline.start || playback.paused || entryStarted) return;
    if (observation.phase === 'idle') {
      observation.phase = 'running';
      observation.lastAt = performance.now();
    } else if (observation.phase === 'complete') {
      entryStarted = true;
      observation.selectedId = null;
    }
    render();
  });
  document.querySelector('#observation-close').addEventListener('click', () => {
    observation.selectedId = null;
    detailView = null;
    recordId = null;
    render();
  });
  function observationText(point) {
    return `움직임 ${point.movement ? '있음' : '없음'} · 사람 추정 ${point.personPossible ? '가능성 있음' : '판단 보류'} · 움직임 정도 ${point.movementLevel}`;
  }
  function openRecord(personId, fromResults = false) {
    const record = records.get(personId);
    if (!record) return;
    observation.selectedId = null;
    detailView = 'record';
    recordId = personId;
    document.querySelector('#record-title').textContent = `${record.region}구역 · ${record.person} · ${record.router}`;
    document.querySelector('#record-observation').textContent = observationText(record.observation);
    document.querySelector('#record-result').textContent = record.result;
    // Populate on selection only: the animation render must not replace typing/IME.
    recordMemo.value = record.memo;
    document.querySelector('#record-back').hidden = !fromResults;
    render();
  }
  function openResults() {
    if (confirmedPeople.size !== layout.people.length) return;
    observation.selectedId = null;
    detailView = 'results';
    recordId = null;
    document.querySelector('#results-count').textContent = `가상 관측 지점 ${observations.size}곳 · 확인된 가상 인원 ${confirmedPeople.size}명`;
    const list = document.querySelector('#results-list');
    list.replaceChildren();
    for (const event of timeline.events) {
      const record = records.get(event.person);
      if (!record) continue;
      const button = document.createElement('button');
      button.type = 'button';
      button.setAttribute('aria-label', `${record.person} 확인 기록 보기`);
      for (const [tag, text] of [
        ['strong', `${record.region}구역 · ${record.person}`],
        ['span', `사전 관측 · ${observationText(record.observation)}`],
        ['span', `확인 · ${record.result}`],
        ...(record.memo ? [['small', `메모 · ${record.memo}`]] : []),
      ]) {
        const line = document.createElement(tag);
        line.textContent = text;
        button.append(line);
      }
      button.addEventListener('click', () => openRecord(record.person, true));
      list.append(button);
    }
    render();
  }
  recordButton.addEventListener('click', () => openRecord(selection?.personId));
  resultsButton.addEventListener('click', openResults);
  document.querySelector('#record-back').addEventListener('click', openResults);
  function updateMemo(event) {
    const record = records.get(recordId);
    if (detailView !== 'record' || !record) return;
    // Let Korean composition finish before enforcing the length limit.
    if (!event.isComposing && recordMemo.value.length > 120) recordMemo.value = recordMemo.value.slice(0, 120);
    record.memo = recordMemo.value.slice(0, 120);
  }
  recordMemo.addEventListener('input', updateMemo);
  recordMemo.addEventListener('compositionend', updateMemo);
  confirmationButton.addEventListener('click', () => {
    const pending = awaitingConfirmation();
    if (!playback.ready || !playback.selected || playback.running || playback.paused || !pending || pending.person !== observation.selectedId) return;
    confirmedPeople.add(pending.person);
    if (!records.has(pending.person)) {
      records.set(pending.person, {
        person: pending.person, region: pending.region, router: pending.router,
        observation: { ...observations.get(pending.person) },
        result: '가상 인원 확인', memo: '',
      });
    }
    observation.selectedId = null;
    // Confirmation changes visibility only. The next route still needs a click.
    // Open once on the last confirmation, never from the animation render.
    if (confirmedPeople.size === layout.people.length) openResults();
    else render();
  });
  function reset() {
    cancelAnimationFrame(animationFrame);
    animationFrame = 0;
    segment = null;
    clearSelection();
    entryStarted = false;
    confirmedPeople.clear();
    records.clear();
    detailView = null;
    recordId = null;
    recordMemo.value = '';
    resultsDetails.open = false;
    document.querySelector('#results-list').replaceChildren();
    Object.assign(observation, { phase: 'idle', elapsed: 0, time: 0, lastAt: null, selectedId: null });
    Object.assign(playback, { time: 0, selected: false, running: false, paused: false });
    building.resetCamera();
    map.reset();
    render();
    status('ready', '지도에서 시연 건물을 선택하세요', '탑뷰에서 사전 관측을 시작할 수 있습니다.');
  }
  pauseButton.addEventListener('click', () => {
    const now = performance.now();
    if (playback.paused) {
      playback.paused = false;
      if (segment) {
        segment.start = playback.time;
        segment.at = now;
        playback.running = true;
      }
      if (selection) selection.lastAt = now;
      observation.lastAt = now;
      render();
      scheduleFrame();
    } else if (playback.running || hasObservationMotion() || selection?.phase === 'playing' || selection?.phase === 'focusing') {
      cancelAnimationFrame(animationFrame);
      animationFrame = 0;
      advance(now);
      playback.paused = Boolean(segment || hasObservationMotion() || selection?.phase === 'playing' || selection?.phase === 'focusing');
      playback.running = false;
      render();
    }
  });
  async function requestReset(replay) {
    if (records.size) {
      const wasPaused = playback.paused;
      cancelAnimationFrame(animationFrame);
      animationFrame = 0;
      if (!wasPaused) advance(performance.now());
      const wasRunning = playback.running;
      playback.paused = true;
      playback.running = false;
      render();
      const dialogAt = performance.now();
      building.pauseCameraTransition();
      const dialog = document.querySelector('#reset-dialog');
      dialog.returnValue = 'cancel';
      const accepted = await new Promise(resolve => {
        dialog.addEventListener('close', () => resolve(dialog.returnValue === 'reset'), { once: true });
        dialog.showModal();
      });
      const now = performance.now();
      building.excludeDialogTime(now - dialogAt);
      if (!accepted) {
        playback.paused = wasPaused;
        playback.running = wasRunning;
        if (segment) { segment.start = playback.time; segment.at = now; }
        if (selection) selection.lastAt = now;
        observation.lastAt = now;
        render();
        return;
      }
    }
    reset();
    if (replay) start();
  }
  document.querySelector('#replay-button').addEventListener('click', () => requestReset(true));
  document.querySelector('#map-button').addEventListener('click', () => requestReset(false));
  function wholeView(mode) {
    clearSelection();
    building.setView(mode);
    render();
  }
  viewButtons.forEach(button => button.addEventListener('click', () => wholeView(button.dataset.view)));
  overviewButton.addEventListener('click', () => wholeView('oblique'));
  const rows = new Map();
  for (const event of timeline.events) {
    const row = document.createElement('li');
    row.dataset.region = event.region;
    const button = document.createElement('button');
    button.type = 'button';
    button.disabled = true;
    button.setAttribute('aria-label', `${event.router} · ${event.person} 구역 진행`);
    const label = document.createElement('strong');
    label.textContent = `${event.router} → ${event.person}`;
    const hint = document.createElement('span');
    button.append(label, hint);
    button.addEventListener('click', () => clickRouter(event.router));
    row.append(button);
    document.querySelector('#region-list').append(row);
    rows.set(event.router, { row, button, hint });
  }
  function render() {
    const time = playback.time;
    const blend = playback.selected ? progress(time, exterior) : 0;
    map.render(progress(time, focus), playback.selected);
    mapLayer.style.opacity = String(1 - blend);
    buildingLayer.style.opacity = String(blend);
    buildingLayer.style.visibility = blend === 0 ? 'hidden' : 'visible';
    mapLayer.style.visibility = blend === 1 ? 'hidden' : 'visible';
    mapLayer.setAttribute('aria-hidden', String(blend === 1));
    buildingLayer.setAttribute('aria-hidden', String(blend === 0));
    const revealState = timeline.at(time, confirmedPeople);
    const next = currentEvent();
    const pending = awaitingConfirmation();
    const canClick = entryStarted && revealState.started && !playback.running && !playback.paused && Boolean(next) && !pending;
    if (revealState.visiblePeople.has(observation.selectedId)) observation.selectedId = null;
    building.render(progress(time, top), revealState, next?.router, canClick, playback.paused, {
      active: observation.phase === 'running', visible: observation.phase === 'complete',
      time: observation.time, selectedId: observation.selectedId,
      posed: observation.selectedId && live.active() ? observation.selectedId : null,
    });
    observationControls.hidden = !revealState.started || entryStarted;
    observationButton.textContent = observation.phase === 'idle' ? '사전 관측' : observation.phase === 'running' ? '가상 관측 중…' : '진입 시작';
    observationButton.disabled = playback.paused || observation.phase === 'running';
    const panelMode = detailView || (observation.selectedId ? 'observation' : null);
    observationPanel.hidden = !panelMode;
    observationPanel.dataset.mode = panelMode || '';
    document.querySelector('#detail-heading').textContent = panelMode === 'record' ? '확인 기록' : panelMode === 'results' ? '시연 결과' : '가상 관측';
    document.querySelector('#observation-content').hidden = panelMode !== 'observation';
    document.querySelector('#record-content').hidden = panelMode !== 'record';
    document.querySelector('#results-content').hidden = panelMode !== 'results';
    confirmationButton.hidden = panelMode !== 'observation' || !pending || pending.person !== observation.selectedId;
    confirmationButton.disabled = playback.paused || playback.running;
    playbackControls.hidden = !playback.selected;
    pauseButton.textContent = playback.paused ? '계속 재생' : '일시정지';
    pauseButton.disabled = !playback.running && !playback.paused && !hasObservationMotion() && !['focusing', 'playing'].includes(selection?.phase);
    pauseButton.setAttribute('aria-pressed', String(playback.paused));
    viewButtons.forEach(button => { button.disabled = !revealState.started; });
    posePanel.hidden = !entryStarted || Boolean(panelMode);
    recordButton.hidden = !records.has(selection?.personId);
    resultsButton.hidden = confirmedPeople.size !== layout.people.length;
    overviewButton.disabled = !selection;
    posePanel.dataset.person = selection?.personId || '';
    const livePlaying = Boolean(selection?.live && selection.phase === 'playing');
    poseEyebrow.textContent = livePlaying ? 'LIVE WI-FI POSE' : 'SYNTHETIC POSE / 5 SEC';
    poseNote.textContent = livePlaying ? '실시간 Wi-Fi CSI 추정 · 이 PC에서 로컬 추론' : '로컬 가상 좌표 샘플 · 실제 Wi-Fi 수신 아님';
    posePanel.dataset.state = selection ? playback.paused && selection.phase === 'playing' ? 'paused' : selection.phase : 'idle';
    poseStatus.textContent = !selection ? '확인된 인원을 클릭하면 실시간 자세 추정을 볼 수 있습니다 (허브 미연결 시 5초 가상 동작)'
      : selection.live && selection.phase === 'playing' ? `${selection.personId} · ${playback.paused ? '일시정지' : '실시간 Wi-Fi 자세 추정'} · ${live.summary()}`
      : `${selection.personId} · ${selection.phase === 'focusing' ? '확대 중' : selection.phase === 'complete' ? '가상 샘플 재생 완료 · 5.0 / 5.0초'
        : `${playback.paused ? '일시정지' : '가상 관절 데이터 수신 중'} · ${(selection.elapsed / 1000).toFixed(1)} / 5.0초`}`;
    document.querySelector('#camera-help').textContent = revealState.started
      ? '드래그 회전 · 스크롤 확대 · 오른쪽 드래그 이동'
      : '내부 화면에 도착하면 시점을 조작할 수 있습니다';
    const countText = `확인된 가상 인원 ${revealState.visiblePeople.size} / ${layout.people.length}명`;
    if (peopleCount.textContent !== countText) peopleCount.textContent = countText;
    for (const region of revealState.regions) {
      const { row, button, hint } = rows.get(region.router);
      const active = playback.running && next?.router === region.router && revealState.started;
      const confirmed = confirmedPeople.has(region.person);
      row.dataset.state = confirmed ? 'complete' : region.complete ? 'confirmation' : active ? 'active' : 'waiting';
      button.disabled = !(canClick && next?.router === region.router);
      hint.textContent = confirmed ? '확인 완료' : region.complete ? '확인 대기' : playback.paused && next?.router === region.router ? '일시정지' : active ? (time < region.signalStart ? '이동 중' : '가상 신호 수신') : !button.disabled ? '클릭하여 진행' : '순서 대기';
    }
    document.querySelector('#reveal-status').hidden = !revealState.started || !entryStarted;
    progressFill.style.transform = `scaleX(${time / timeline.end})`;
    // Keep only the decorative pulses moving during manual waits/completion.
    // advance() still changes the playhead only while an existing segment runs.
    scheduleFrame();
    if (!playback.selected) return;
    if (playback.paused) {
      routeTarget.textContent = '시연 일시정지';
      status('paused', '시연을 일시정지했습니다', revealState.started ? '시점은 자유롭게 조작할 수 있습니다. 계속 재생하면 멈춘 지점부터 이어집니다.' : '계속 재생하면 멈춘 화면 전환부터 이어집니다.', revealState.started ? (entryStarted ? 'reveal' : 'top') : time >= top.start ? 'top' : time >= exterior.start ? 'exterior' : 'map');
    } else if (revealState.started && !entryStarted) {
      const observing = observation.phase === 'running';
      const observed = observation.phase === 'complete';
      status(`observation-${observation.phase}`, observing ? '가상 관측 중' : observed ? '가상 관측 완료' : '진입 전 가상 관측',
        observing ? 'R1~R4의 가상 신호를 표시합니다.' : observed ? '점멸 지점을 선택해 정보를 보고, 준비되면 진입 시작을 누르세요.' : '사전 관측을 눌러 내부의 가상 관측 지점을 살펴보세요.', 'top');
    } else if (pending) {
      routeTarget.textContent = `${pending.region}구역 확인 대기`;
      status(`confirmation-${pending.person}`, `${pending.region}구역 · 확인 대기`, '점멸 지점을 선택해 확인하세요.', 'reveal');
    } else if (confirmedPeople.size === layout.people.length) {
      routeTarget.textContent = '전체 지점 확인 완료';
      status('complete', '전체 지점 확인 완료', '가상 라우터 4개 · 확인된 인원 4명 · 지나온 동선을 유지합니다.', 'reveal');
    } else if (canClick) {
      routeTarget.textContent = `${next.router} 클릭 → ${next.person} 구역 이동`;
      status(`waiting-${next.router}`, `${next.router} 클릭을 기다립니다`, '도면의 라우터 또는 위쪽 목록을 누르면 다음 구역으로 진행합니다.', 'reveal');
    } else if (revealState.started && next) {
      const moving = time < next.signalStart;
      routeTarget.textContent = moving ? `${next.person} 구역으로 이동 중` : `${next.router} 가상 신호 수신`;
      status(`${moving ? 'moving' : 'reveal'}-${next.router}`, moving ? `${next.person} 구역으로 이동합니다` : `${next.router} 가상 신호 수신 · ${next.region}구역 공개`, moving ? '지나온 동선은 초록색 점선으로 남습니다.' : '공개 후 점멸 지점을 선택해 확인할 수 있습니다.', 'reveal');
    } else if (time >= top.start) {
      status('top', '위에서 보는 화면으로 이동합니다', '탑뷰 전환 후 사전 관측을 시작할 수 있습니다.', 'top');
    } else if (time >= exterior.start) {
      status('exterior', 'ANSEN 예시 모델로 전환합니다', '실내를 가린 단순 외피를 표시합니다.', 'exterior');
    } else {
      status('map-focus', '선택한 장소로 이동합니다', '주소 참조점을 확대하고 있습니다.');
    }
  }
  function advance(now) {
    if (playback.running && segment) {
      playback.time = Math.min(segment.end, segment.start + Math.max(0, now - segment.at) / 1000);
      playback.running = playback.time < segment.end;
      if (!playback.running) segment = null;
    }
    // This clock only drives synthetic observation and blinking. It never
    // advances the existing entry/reveal playhead while the user is reviewing.
    if (!playback.paused && observation.phase !== 'idle') {
      const delta = observation.lastAt === null ? 0 : Math.max(0, now - observation.lastAt) / 1000;
      observation.lastAt = now;
      observation.time += delta;
      if (observation.phase === 'running') {
        observation.elapsed = Math.min(observationSample.durationSeconds, observation.elapsed + delta);
        if (observation.elapsed >= observationSample.durationSeconds) observation.phase = 'complete';
      }
    }
    const observed = observation.selectedId;
    if (observation.posedId && observation.posedId !== observed) building.resetPose(observation.posedId);
    observation.posedId = observed;
    if (observed && !playback.paused && live.step(Math.max(0, now - (observation.poseAt ?? now)), sampledJoints)) {
      building.updatePose(observed, sampledJoints);
    }
    observation.poseAt = now;
    if (!playback.paused && selection?.phase === 'playing') {
      const delta = Math.max(0, now - selection.lastAt);
      selection.lastAt = now;
      const liveJoints = live.step(delta, sampledJoints);
      selection.live = Boolean(liveJoints);
      if (liveJoints) building.updatePose(selection.personId, liveJoints);
      else {
        selection.elapsed = Math.min(poseSample.durationMs, selection.elapsed + delta);
        building.updatePose(selection.personId, samplePose(poseSample, selection.elapsed, sampledJoints));
      }
      if (!selection.live && selection.elapsed === poseSample.durationMs) {
        building.resetPose(selection.personId);
        selection.phase = 'complete';
        building.setView('oblique');
      }
    }
  }
  function frame(now) {
    animationFrame = 0;
    if (playback.paused) return;
    advance(now);
    render();
    scheduleFrame();
  }
  // Resizing changes the viewport without advancing the paused playhead.
  const resizeObserver = new ResizeObserver(() => {
    map.resize();
    if (building) { building.resize(); render(); }
  });
  resizeObserver.observe(document.querySelector('#stage'));
  resizeObserver.observe(document.querySelector('#building-canvas'));
  building = await createBuilding(document.querySelector('#building-canvas'), settings.model, event => {
    detail.textContent = event.total && event.loaded < event.total
      ? `ANSEN 모델 불러오는 중… ${Math.round(event.loaded / event.total * 100)}% · 준비 후 선택할 수 있습니다.`
      : 'ANSEN 모델의 화면을 준비하고 있습니다…';
  }, { layout, pose, timeline, onRouterClick: clickRouter, onPersonClick: selectPerson, onObservationClick: selectObservation, onViewChange(mode) {
    viewButtons.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.view === mode)));
  } });
  render();
  playback.ready = true;
  map.setReady();
  status('ready', '지도에서 시연 건물을 선택하세요', '탑뷰에서 사전 관측을 시작할 수 있습니다.');
}

init().catch(error => {
  playback.ready = false;
  status('error', '시연 준비에 실패했습니다', `불러오기 오류: ${error.message || String(error)}`);
  console.error(error);
});
