import * as THREE from 'three';
import { loadModel } from './load-model.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { prepareOccupants } from './preparation/mock-occupants.js?v=5';
import { createEntryView } from './entry-view.js?v=4';

export async function createBuilding(container, modelUrl, onProgress, { layout, pose, timeline, onRouterClick, onViewChange, onPersonClick, onObservationClick }) {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setClearColor(0x070a08);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1;
  renderer.domElement.setAttribute('aria-label', '가상 신호에 맞춰 순차 공개하는 ANSEN 예시 모델');
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const ambient = new THREE.HemisphereLight(0xe5f1ff, 0x263640, 2.5);
  scene.add(ambient);
  const sun = new THREE.DirectionalLight(0xffffff, 3);
  sun.position.set(-12, 25, 18);
  scene.add(sun);
  const camera = new THREE.OrthographicCamera(-20, 20, 15, -15, .01, 500);
  // The intro follows the playhead; user camera controls become independent afterward.
  const gltf = await loadModel(modelUrl, onProgress);
  const interior = gltf.scene;
  // Hide BEFORE adding or rendering. The cover remains fully closed until the
  // top view, and every person group is gated separately from the cover.
  interior.visible = false;
  const bounds = new THREE.Box3().setFromObject(interior);
  scene.add(interior);
  const center = bounds.getCenter(new THREE.Vector3());
  const size = bounds.getSize(new THREE.Vector3());
  // Open only two movable door leaves in this view; the source model and
  // preparation viewer stay intact. Fixed route points pass through openings.
  for (const [names, x, z, angle] of [
    [/^mesh_139[1-4]_/, 4.368, -5.072, -Math.PI / 2],
    [/^mesh_140[6-9]_/, 2.468, -5.390, Math.PI / 2],
  ]) {
    const leaves = [];
    interior.traverse(object => { if (object.isMesh && names.test(object.name)) leaves.push(object); });
    const hinge = new THREE.Group();
    hinge.position.set(x, 0, z);
    scene.add(hinge);
    hinge.updateMatrixWorld();
    for (const leaf of leaves) hinge.attach(leaf);
    hinge.rotation.y = angle;
    // Reparent under the original interior so its visibility also gates doors.
    interior.attach(hinge);
  }
  // The GLB root already converts millimetres to metres. Do not rescale it.
  const extent = Math.max(size.x, size.y, size.z);
  const coverSize = size.clone().addScalar(.12);
  const coverGeometry = new THREE.BoxGeometry(coverSize.x, coverSize.y, coverSize.z);
  const side = new THREE.MeshStandardMaterial({ color: 0x202b24, roughness: 1 });
  const roof = new THREE.MeshStandardMaterial({ color: 0x0e1812, roughness: 1 });
  const cover = new THREE.Mesh(coverGeometry, [side, side, roof, side, side, side]);
  cover.position.copy(center);
  scene.add(cover);
  const outline = new THREE.LineSegments(new THREE.EdgesGeometry(coverGeometry), new THREE.LineBasicMaterial({ color: 0x516d58 }));
  outline.position.copy(center);
  scene.add(outline);

  // A faint floor-plan drawing of architectural walls only. It stays on the
  // floor plane when the user tilts the camera; no furniture is exposed.
  const wallPreview = new THREE.Group();
  const overlayY = .275;
  interior.updateMatrixWorld(true);
  interior.traverse(object => {
    if (!object.isMesh || !/^mesh_142[3-5]_/.test(object.name)) return;
    const geometry = new THREE.EdgesGeometry(object.geometry, 20);
    geometry.applyMatrix4(object.matrixWorld);
    const positions = geometry.attributes.position;
    for (let i = 0; i < positions.count; i++) positions.setY(i, overlayY);
    positions.needsUpdate = true;
    const lines = new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({
      color: 0x81978a, transparent: true, opacity: .25, depthTest: false, depthWrite: false,
    }));
    lines.renderOrder = 20;
    wallPreview.add(lines);
  });
  wallPreview.visible = false;
  scene.add(wallPreview);
  const entryView = createEntryView({ scene, camera, container, layout, onRouterClick });

  const revealUniforms = {
    demoProgress: { value: timeline.events.map(() => 0) },
    demoRegions: { value: timeline.events.map(event => new THREE.Vector4(...event.bounds.x, ...event.bounds.z)) },
    demoOrigins: { value: timeline.events.map(event => new THREE.Vector2(event.origin[0], event.origin[2])) },
    demoRadii: { value: timeline.events.map(event => event.radius) },
    demoLimits: { value: new THREE.Vector4(
      Math.min(...layout.regions.map(region => region.x[0])), Math.max(...layout.regions.map(region => region.x[1])),
      Math.min(...layout.regions.map(region => region.z[0])), Math.max(...layout.regions.map(region => region.z[1])),
    ) },
  };
  // Apply one world-space reveal condition to both the cover and the model.
  // In an oblique view, looking through an open roof cannot expose geometry
  // from a later region. Materials/textures in the preparation viewer are untouched.
  const interiorMaterials = new Map();
  interior.traverse(object => {
    if (!object.isMesh) return;
    const cloneMaterial = source => {
      if (!interiorMaterials.has(source)) interiorMaterials.set(source, source.clone());
      return interiorMaterials.get(source);
    };
    object.material = Array.isArray(object.material) ? object.material.map(cloneMaterial) : cloneMaterial(object.material);
  });
  const modelMaterials = new Set(interiorMaterials.values());
  for (const material of [side, roof, ...wallPreview.children.map(line => line.material), ...modelMaterials]) {
    const isPreview = material.isLineBasicMaterial;
    const isInterior = modelMaterials.has(material);
    material.onBeforeCompile = shader => {
      Object.assign(shader.uniforms, revealUniforms);
      shader.vertexShader = `varying vec2 demoXZ;\n${shader.vertexShader}`.replace(
        '#include <begin_vertex>', '#include <begin_vertex>\ndemoXZ = (modelMatrix * vec4(position, 1.0)).xz;',
      );
      shader.fragmentShader = `
        varying vec2 demoXZ;
        uniform float demoProgress[${timeline.events.length}];
        uniform vec4 demoRegions[${timeline.events.length}];
        uniform vec2 demoOrigins[${timeline.events.length}];
        uniform float demoRadii[${timeline.events.length}];
        uniform vec4 demoLimits;
      ${shader.fragmentShader}`.replace('#include <color_fragment>', `
        #include <color_fragment>
        vec2 point = clamp(demoXZ, demoLimits.xz, demoLimits.yw);
        bool revealed = false;
        float band = 0.0;
        for (int i = 0; i < ${timeline.events.length}; i++) {
          vec4 region = demoRegions[i];
          if (point.x >= region.x && point.x <= region.y && point.y >= region.z && point.y <= region.w) {
            float progress = demoProgress[i];
            if (progress >= 1.0) revealed = true;
            if (progress > 0.0) {
              float edge = distance(point, demoOrigins[i]) - demoRadii[i] * progress;
              if (edge < 0.0) revealed = true;
              band = max(band, 1.0 - smoothstep(0.0, 0.32, edge));
            }
          }
        }
        ${isInterior ? 'if (!revealed) discard;' : `
          if (revealed) discard;
          diffuseColor.rgb = mix(diffuseColor.rgb, vec3(0.175, 0.672, 0.223), band * 0.7);
        `}
      `);
    };
    material.customProgramCacheKey = () => `demo-${isInterior ? 'interior' : isPreview ? 'outline' : 'cover'}-${timeline.events.length}`;
  }

  // One restrained ripple follows the existing reveal fraction, clipped to the
  // demonstration region. It has no clock or radio-range model of its own.
  const signalWaves = timeline.events.map(event => {
    const { x, z } = event.bounds;
    const material = new THREE.ShaderMaterial({
      transparent: true, depthTest: false, depthWrite: false, toneMapped: false,
      uniforms: {
        signalProgress: { value: 0 },
        signalOrigin: { value: new THREE.Vector2(event.origin[0], event.origin[2]) },
        signalRadius: { value: event.radius },
        signalColor: { value: new THREE.Color(0x74d682) },
      },
      vertexShader: `
        varying vec2 signalXZ;
        void main() {
          vec4 world = modelMatrix * vec4(position, 1.0);
          signalXZ = world.xz;
          gl_Position = projectionMatrix * viewMatrix * world;
        }
      `,
      fragmentShader: `
        varying vec2 signalXZ;
        uniform float signalProgress;
        uniform vec2 signalOrigin;
        uniform float signalRadius;
        uniform vec3 signalColor;
        void main() {
          float radius = signalRadius * signalProgress;
          float distanceToOrigin = distance(signalXZ, signalOrigin);
          float front = 1.0 - smoothstep(0.025, 0.11, abs(distanceToOrigin - radius));
          float echo = (1.0 - smoothstep(0.015, 0.08, abs(distanceToOrigin - max(0.0, radius - 0.45)))) * 0.32;
          float fade = smoothstep(0.0, 0.08, signalProgress) * (1.0 - smoothstep(0.75, 1.0, signalProgress));
          float alpha = max(front, echo) * fade * 0.65;
          if (alpha < 0.005) discard;
          gl_FragColor = vec4(signalColor, alpha);
          #include <colorspace_fragment>
        }
      `,
    });
    const wave = new THREE.Mesh(new THREE.PlaneGeometry(x[1] - x[0], z[1] - z[0]), material);
    wave.rotation.x = -Math.PI / 2;
    wave.position.set((x[0] + x[1]) / 2, event.origin[1] + .035, (z[0] + z[1]) / 2);
    wave.name = `${event.router} demonstration signal ripple`;
    wave.renderOrder = 24;
    wave.visible = false;
    scene.add(wave);
    return wave;
  });

  // Reuse these small emissions during pre-entry observation; afterward each
  // starts only when its region is revealed, without advancing the demonstration.
  const routerPulses = timeline.events.map(event => {
    const material = new THREE.ShaderMaterial({
      transparent: true, depthTest: false, depthWrite: false, toneMapped: false,
      uniforms: {
        signalTime: { value: 0 },
        signalColor: { value: new THREE.Color(0xb5ff96) },
      },
      vertexShader: `
        varying vec2 pulsePosition;
        void main() {
          pulsePosition = position.xy;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        varying vec2 pulsePosition;
        uniform float signalTime;
        uniform vec3 signalColor;
        float pulse(float phase) {
          float radius = mix(0.20, 0.95, phase);
          float edge = abs(length(pulsePosition) - radius);
          float line = 1.0 - smoothstep(0.015, 0.045, edge);
          float glow = (1.0 - smoothstep(0.025, 0.12, edge)) * 0.18;
          float fade = smoothstep(0.0, 0.15, phase) * (1.0 - smoothstep(0.45, 1.0, phase));
          return (line + glow) * fade;
        }
        void main() {
          float cycle = signalTime / 0.9;
          float alpha = pulse(fract(cycle));
          if (cycle >= 0.5) alpha = max(alpha, pulse(fract(cycle - 0.5)));
          alpha *= smoothstep(0.0, 0.12, signalTime) * 0.48;
          if (alpha < 0.005) discard;
          gl_FragColor = vec4(signalColor, alpha);
          #include <colorspace_fragment>
        }
      `,
    });
    const pulse = new THREE.Mesh(new THREE.PlaneGeometry(2.2, 2.2), material);
    pulse.rotation.x = -Math.PI / 2;
    pulse.position.set(event.origin[0], event.origin[1] + .04, event.origin[2]);
    pulse.name = `${event.router} decorative emission pulse`;
    pulse.renderOrder = 25;
    pulse.visible = false;
    scene.add(pulse);
    return pulse;
  });
  let pulseLastAt = null;

  const occupants = await prepareOccupants({ layout, pose, visualStyle: 'demo' });
  const people = new Map(layout.people.map(person => [person.id, occupants.group.getObjectByName(person.id)]));
  for (const body of people.values()) {
    body.visible = false;
    // Accessible screen labels replace only this view's name sprites.
    body.traverse(object => {
      if (object.isSprite) object.visible = false;
      // Transparent interior surfaces render after the opaque pass. Keep the
      // demo skeleton in that same pass so its higher renderOrder stays above
      // glass/furnishings, including when the camera focuses on one person.
      if (object.isMesh) { object.material.transparent = true; object.material.opacity = 1; }
    });
  }
  const personPins = document.createElement('div');
  personPins.className = 'person-pins';
  personPins.hidden = true;
  personPins.setAttribute('role', 'group');
  personPins.setAttribute('aria-label', '공개된 가상 인원 선택');
  container.appendChild(personPins);
  const personButtons = new Map(layout.people.map(person => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'person-pin';
    button.dataset.person = person.id;
    button.textContent = person.id;
    button.hidden = true;
    button.setAttribute('aria-label', `${person.id} 스켈레톤 선택 · 5초 가상 동작`);
    button.setAttribute('aria-pressed', 'false');
    let start = null, dragged = false;
    button.addEventListener('pointerdown', event => {
      event.stopPropagation();
      start = [event.clientX, event.clientY];
      dragged = event.button !== 0;
      button.setPointerCapture(event.pointerId);
    });
    button.addEventListener('pointermove', event => {
      event.stopPropagation();
      if (start && Math.hypot(event.clientX - start[0], event.clientY - start[1]) > 5) dragged = true;
    });
    button.addEventListener('pointerup', event => { event.stopPropagation(); start = null; });
    button.addEventListener('pointercancel', () => { start = null; dragged = true; });
    button.addEventListener('contextmenu', event => event.preventDefault());
    button.addEventListener('click', event => {
      event.stopPropagation();
      if (latest.state.visiblePeople.has(person.id) && (event.detail === 0 || !dragged)) onPersonClick?.(person.id);
    });
    personPins.appendChild(button);
    return [person.id, button];
  }));
  const observationPins = document.createElement('div');
  observationPins.className = 'observation-pins';
  observationPins.hidden = true;
  observationPins.setAttribute('role', 'group');
  observationPins.setAttribute('aria-label', '가상 관측 지점 선택');
  container.appendChild(observationPins);
  const observationButtons = new Map(layout.people.map(person => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'observation-pin';
    button.dataset.person = person.id;
    button.hidden = true;
    button.setAttribute('aria-label', `${person.region}구역 가상 관측 지점`);
    button.setAttribute('aria-pressed', 'false');
    let start = null, dragged = false;
    button.addEventListener('pointerdown', event => {
      event.stopPropagation();
      start = [event.clientX, event.clientY];
      dragged = event.button !== 0;
      button.setPointerCapture(event.pointerId);
    });
    button.addEventListener('pointermove', event => {
      event.stopPropagation();
      if (start && Math.hypot(event.clientX - start[0], event.clientY - start[1]) > 5) dragged = true;
    });
    button.addEventListener('pointerup', event => { event.stopPropagation(); start = null; });
    button.addEventListener('pointercancel', () => { start = null; dragged = true; });
    button.addEventListener('contextmenu', event => event.preventDefault());
    button.addEventListener('click', event => {
      event.stopPropagation();
      if (latest.observation.visible && !latest.state.visiblePeople.has(person.id) && (event.detail === 0 || !dragged)) {
        onObservationClick?.(person.id);
      }
    });
    observationPins.appendChild(button);
    return [person.id, button];
  }));
  const routers = new Map(layout.routers.map(router => {
    const node = occupants.group.getObjectByName(router.id);
    const materials = [];
    node.traverse(object => {
      if (!object.material) return;
      if (object.isSprite) { object.visible = false; return; } // DOM button carries the router name.
      object.material = object.material.clone(); // Preparation routers share their base material.
      object.material.color.set(0x74d682);
      object.material.toneMapped = false;
      object.material.transparent = true;
      materials.push(object.material);
    });
    return [router.id, materials];
  }));
  occupants.group.visible = false;
  scene.add(occupants.group);

  function setReveal(state) {
    // The supplied model includes downward-facing surfaces. Restore the
    // preparation viewer's ground lighting so revealed floors/walls stay legible.
    ambient.color.set(state.started ? 0xffffff : 0xe5f1ff);
    ambient.groundColor.set(state.started ? 0xa6a8a3 : 0x263640);
    ambient.intensity = state.started ? 2 : 2.5;
    sun.intensity = state.started ? 2 : 3;
    interior.visible = state.started;
    cover.visible = !state.complete;
    outline.visible = !state.complete;
    wallPreview.visible = state.started && !state.complete;
    occupants.group.visible = state.started;
    state.regions.forEach((region, index) => {
      revealUniforms.demoProgress.value[index] = region.progress;
      const active = region.progress > 0;
      for (const material of routers.get(region.router)) material.opacity = region.complete || active ? 1 : .28;
      const wave = signalWaves[index];
      wave.material.uniforms.signalProgress.value = region.progress;
      wave.visible = state.started && region.progress > 0 && region.progress < 1;
    });
    for (const [id, body] of people) body.visible = state.visiblePeople.has(id);
  }

  // Y remains the orbit axis. A tiny Z offset gives the overhead preset the
  // same +X right / -Z up orientation without a singular lookAt at the pole.
  camera.up.set(0, 1, 0);
  const obliqueOffset = new THREE.Vector3(extent * .12, extent * .9, extent);
  const topOffset = new THREE.Vector3(0, extent * 2, extent * .000002);
  camera.position.copy(center).add(obliqueOffset);
  camera.lookAt(center);
  const obliqueRotation = camera.quaternion.clone();
  camera.position.copy(center).add(topOffset);
  camera.lookAt(center);
  const topRotation = camera.quaternion.clone();
  camera.far = extent * 10;
  const corners = [];
  for (const x of [-1, 1]) for (const y of [-1, 1]) for (const z of [-1, 1]) {
    corners.push(new THREE.Vector3(x * coverSize.x / 2, y * coverSize.y / 2, z * coverSize.z / 2));
  }
  const inverse = new THREE.Quaternion();
  const projected = new THREE.Vector3();
  let aspect = 1, halfSpan = extent;
  let cameraInteractive = false;
  let transition = null, cameraFrame = 0;
  let suppressControlDraw = false;
  const noObservation = { active: false, visible: false, time: 0, selectedId: null };
  let latest = { topProgress: 0, state: timeline.at(0), nextRouter: undefined, enabled: false, observation: noObservation };
  let viewMode = 'intro', selectedPerson = null, focusSettled = false;
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enabled = false;
  // Direct interaction needs no permanent rendering loop or residual motion.
  controls.enableDamping = false;
  controls.minPolarAngle = .000001;
  controls.maxPolarAngle = Math.PI / 2 - .06;
  controls.minZoom = .45;
  controls.maxZoom = 5;
  controls.target.copy(center);
  controls.cursor.copy(center);
  controls.maxTargetRadius = extent;
  controls.mouseButtons.LEFT = THREE.MOUSE.ROTATE;
  controls.mouseButtons.MIDDLE = THREE.MOUSE.DOLLY;
  controls.mouseButtons.RIGHT = THREE.MOUSE.PAN;

  function setMode(mode) {
    if (viewMode === mode) return;
    viewMode = mode;
    renderer.domElement.dataset.view = mode;
    onViewChange?.(mode);
  }
  function fitSpan(rotation) {
    inverse.copy(rotation).invert();
    let halfWidth = 0, halfHeight = 0;
    for (const corner of corners) {
      projected.copy(corner).applyQuaternion(inverse);
      halfWidth = Math.max(halfWidth, Math.abs(projected.x));
      halfHeight = Math.max(halfHeight, Math.abs(projected.y));
    }
    return Math.max(halfHeight, halfWidth / aspect) * 1.12;
  }
  function updateProjection() {
    camera.left = -halfSpan * aspect;
    camera.right = halfSpan * aspect;
    camera.top = halfSpan;
    camera.bottom = -halfSpan;
    camera.updateProjectionMatrix();
  }
  const personPoint = new THREE.Vector3();
  const cameraPoint = new THREE.Vector3();
  const routerButtons = [...container.querySelectorAll('.router-pin')];
  function draw() {
    entryView.render(latest.state, latest.nextRouter, latest.enabled);
    personPins.hidden = !latest.state.started;
    camera.updateMatrixWorld();
    for (const [id, body] of people) {
      const button = personButtons.get(id);
      button.setAttribute('aria-pressed', String(selectedPerson === id));
      if (!body.visible || !latest.state.visiblePeople.has(id)) { button.hidden = true; continue; }
      body.updateWorldMatrix(true, false);
      personPoint.set(0, 1.68, 0).applyMatrix4(body.matrixWorld);
      cameraPoint.copy(personPoint).applyMatrix4(camera.matrixWorldInverse);
      personPoint.project(camera);
      button.hidden = cameraPoint.z >= 0 || Math.abs(personPoint.x) > 1 || Math.abs(personPoint.y) > 1 || Math.abs(personPoint.z) > 1;
      button.style.left = `${(personPoint.x * .5 + .5) * container.clientWidth}px`;
      button.style.top = `${(-personPoint.y * .5 + .5) * container.clientHeight}px`;
    }
    observationPins.hidden = !latest.state.started || !latest.observation.visible;
    for (const person of layout.people) {
      const button = observationButtons.get(person.id);
      button.setAttribute('aria-pressed', String(latest.observation.selectedId === person.id));
      if (observationPins.hidden || latest.state.visiblePeople.has(person.id)) { button.hidden = true; continue; }
      // Project the existing floor anchor, not a label offset or a new location.
      // Only this small dot overlays the cover; no model materials are changed.
      personPoint.fromArray(person.position);
      personPoint.y += .035;
      cameraPoint.copy(personPoint).applyMatrix4(camera.matrixWorldInverse);
      personPoint.project(camera);
      button.hidden = cameraPoint.z >= 0 || Math.abs(personPoint.x) > 1 || Math.abs(personPoint.y) > 1 || Math.abs(personPoint.z) > 1;
      button.style.left = `${(personPoint.x * .5 + .5) * container.clientWidth}px`;
      button.style.top = `${(-personPoint.y * .5 + .5) * container.clientHeight}px`;
      button.style.setProperty('--observation-opacity', String(.65 + .25 * Math.sin(latest.observation.time * Math.PI * 2 / 1.8)));
    }
    // A focused person's arms should remain visible behind nearby router
    // labels. Only the screen label moves; the router and its world anchor stay.
    const focusedBody = focusSettled && people.get(selectedPerson);
    if (focusedBody?.visible) {
      const width = container.clientWidth, height = container.clientHeight;
      let left = Infinity, right = -Infinity, top = Infinity, bottom = -Infinity;
      for (const joint of focusedBody.children) {
        if (joint.userData.kind !== 'skeleton-joint') continue;
        joint.getWorldPosition(personPoint);
        cameraPoint.copy(personPoint).applyMatrix4(camera.matrixWorldInverse);
        personPoint.project(camera);
        if (cameraPoint.z >= 0 || Math.abs(personPoint.z) > 1) continue;
        const x = (personPoint.x * .5 + .5) * width, y = (-personPoint.y * .5 + .5) * height;
        left = Math.min(left, x - 10); right = Math.max(right, x + 10);
        top = Math.min(top, y - 10); bottom = Math.max(bottom, y + 10);
      }
      for (const button of routerButtons) {
        if (button.hidden) continue;
        const w = button.offsetWidth, h = button.offsetHeight;
        const x = parseFloat(button.style.left), y = parseFloat(button.style.top);
        // Router labels use translate(-50%, -140%) in the existing stylesheet.
        if (x + w / 2 <= left || x - w / 2 >= right || y - h * .4 <= top || y - h * 1.4 >= bottom) continue;
        const candidates = [left - w / 2, right + w / 2];
        const fitting = candidates.filter(value => value >= w / 2 + 2 && value <= width - w / 2 - 2);
        const target = (fitting.length ? fitting : candidates).sort((a, b) => Math.abs(a - x) - Math.abs(b - x))[0];
        button.style.left = `${Math.max(w / 2 + 2, Math.min(width - w / 2 - 2, target))}px`;
      }
    }
    renderer.render(scene, camera);
  }
  function stopTransition() {
    cancelAnimationFrame(cameraFrame);
    cameraFrame = 0;
    transition = null;
  }
  function interruptCamera() {
    if (!cameraInteractive) return;
    const ready = transition?.onReady;
    stopTransition();
    if (ready) focusSettled = true;
    setMode('free');
    ready?.();
  }
  controls.addEventListener('start', () => {
    if (!cameraInteractive) return;
    // OrbitControls also emits start on a simple pointer down. A focus tween
    // ends only on an actual drag/wheel, not before a person click is resolved.
    if (!transition?.onReady) { stopTransition(); setMode('free'); }
  });
  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  let pointerStart = null, canvasDragged = false;
  renderer.domElement.addEventListener('pointerdown', event => {
    pointerStart = [event.clientX, event.clientY];
    canvasDragged = event.button !== 0 || !event.isPrimary;
  }, true);
  renderer.domElement.addEventListener('pointermove', event => {
    if (pointerStart && Math.hypot(event.clientX - pointerStart[0], event.clientY - pointerStart[1]) > 5) {
      canvasDragged = true;
      interruptCamera();
    }
  }, true);
  renderer.domElement.addEventListener('pointerup', () => { pointerStart = null; }, true);
  renderer.domElement.addEventListener('pointercancel', () => { pointerStart = null; canvasDragged = true; }, true);
  renderer.domElement.addEventListener('wheel', interruptCamera, { capture: true, passive: true });
  renderer.domElement.addEventListener('click', event => {
    if (!cameraInteractive || event.button !== 0 || canvasDragged) return;
    const rect = renderer.domElement.getBoundingClientRect();
    pointer.set((event.clientX - rect.left) / rect.width * 2 - 1, -(event.clientY - rect.top) / rect.height * 2 + 1);
    raycaster.setFromCamera(pointer, camera);
    const pickable = [];
    for (const [id, body] of people) {
      if (body.visible && latest.state.visiblePeople.has(id)) {
        body.traverse(object => { if (object.isMesh && object.visible) pickable.push(object); });
      }
    }
    const hit = raycaster.intersectObjects(pickable, false)[0];
    if (hit) { onPersonClick?.(hit.object.parent.name); return; }
    // An eight-pixel joint target keeps tiny top-view skeletons selectable.
    let nearest = null, distance = 8;
    for (const object of pickable) {
      if (object.userData.kind !== 'skeleton-joint') continue;
      object.getWorldPosition(personPoint);
      cameraPoint.copy(personPoint).applyMatrix4(camera.matrixWorldInverse);
      personPoint.project(camera);
      if (cameraPoint.z >= 0 || Math.abs(personPoint.z) > 1) continue;
      const pixels = Math.hypot((personPoint.x - pointer.x) * rect.width / 2, (personPoint.y - pointer.y) * rect.height / 2);
      if (pixels < distance) { distance = pixels; nearest = object.parent.name; }
    }
    if (nearest) onPersonClick?.(nearest);
  });
  controls.addEventListener('change', () => {
    if (cameraInteractive && !suppressControlDraw) draw();
  });
  function introCamera(progress) {
    camera.quaternion.slerpQuaternions(obliqueRotation, topRotation, progress);
    const distance = THREE.MathUtils.lerp(obliqueOffset.length(), topOffset.length(), progress);
    camera.position.set(0, 0, distance).applyQuaternion(camera.quaternion).add(center);
    camera.zoom = 1;
    controls.target.copy(center);
    halfSpan = fitSpan(camera.quaternion);
    updateProjection();
  }
  function animateView(now) {
    cameraFrame = 0;
    if (!transition || !cameraInteractive) return;
    const t = Math.min((now - transition.at) / 700, 1);
    const smooth = t * t * (3 - 2 * t);
    controls.target.lerpVectors(transition.fromTarget, transition.toTarget, smooth);
    const distance = THREE.MathUtils.lerp(transition.fromDistance, transition.toDistance, smooth);
    const phi = THREE.MathUtils.lerp(transition.fromPhi, transition.toPhi, smooth);
    const theta = transition.fromTheta + transition.thetaDelta * smooth;
    camera.position.setFromSphericalCoords(distance, phi, theta).add(controls.target);
    camera.lookAt(controls.target);
    camera.zoom = THREE.MathUtils.lerp(transition.fromZoom, 1, smooth);
    const targetSpan = transition.personId ? personSpan() : fitSpan(transition.toRotation);
    halfSpan = THREE.MathUtils.lerp(transition.fromSpan, targetSpan, smooth);
    updateProjection();
    const ready = t === 1 ? transition.onReady : null;
    if (t === 1) {
      if (transition.personId) focusSettled = true;
      transition = null;
      suppressControlDraw = true;
      controls.update();
      suppressControlDraw = false;
    } else {
      cameraFrame = requestAnimationFrame(animateView);
    }
    draw();
    ready?.();
  }
  function personSpan() {
    return Math.max(1.2, 1.15 / aspect);
  }
  function startView(target, offset, rotation, mode, personId = null, onReady = null) {
    stopTransition();
    const from = new THREE.Spherical().setFromVector3(camera.position.clone().sub(controls.target));
    const to = new THREE.Spherical().setFromVector3(offset);
    transition = {
      at: performance.now(),
      fromPhi: from.phi, toPhi: to.phi, fromTheta: from.theta,
      thetaDelta: Math.atan2(Math.sin(to.theta - from.theta), Math.cos(to.theta - from.theta)),
      toRotation: rotation.clone(), fromTarget: controls.target.clone(), toTarget: target.clone(),
      fromDistance: camera.position.distanceTo(controls.target), toDistance: offset.length(),
      fromZoom: camera.zoom, fromSpan: halfSpan, personId, onReady,
    };
    setMode(mode);
    cameraFrame = requestAnimationFrame(animateView);
  }
  function highlightPerson(id) {
    selectedPerson = id;
    for (const [personId, body] of people) {
      body.traverse(object => {
        if (!object.isMesh) return;
        const joint = object.userData.kind === 'skeleton-joint';
        object.material.color.set(personId === id ? (joint ? 0xffffff : 0xcaff99) : (joint ? 0xe8ffec : 0x74d682));
      });
    }
  }
  function focusPerson(id, onReady) {
    const body = people.get(id);
    if (!cameraInteractive || !body?.visible || !latest.state.visiblePeople.has(id)) return false;
    if (id === selectedPerson && focusSettled) {
      stopTransition();
      onReady?.();
      return true;
    }
    highlightPerson(id);
    focusSettled = false;
    const target = body.position.clone().add(new THREE.Vector3(0, .9, 0));
    const offset = new THREE.Vector3(2.4, 3.2, 6).normalize().multiplyScalar(extent * 2);
    offset.applyAxisAngle(camera.up, body.rotation.y);
    const rotation = new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().lookAt(target.clone().add(offset), target, camera.up));
    startView(target, offset, rotation, 'person', id, onReady);
    draw();
    return true;
  }
  function clearPersonSelection() {
    stopTransition(); // Discard the previous onReady without starting its sample.
    highlightPerson(null);
    focusSettled = false;
  }
  function setView(mode) {
    if (!cameraInteractive || !['top', 'oblique'].includes(mode)) return;
    clearPersonSelection();
    startView(center, mode === 'top' ? topOffset : obliqueOffset, mode === 'top' ? topRotation : obliqueRotation, mode);
  }
  function resetCamera() {
    clearPersonSelection();
    cameraInteractive = false;
    controls.enabled = false;
    // reset also ends any in-progress mouse gesture; damping is disabled.
    controls.reset();
    introCamera(0);
    renderer.domElement.dataset.cameraReady = 'false';
    setMode('intro');
  }
  function resize() {
    const { width, height } = container.getBoundingClientRect();
    const oldAspect = aspect;
    aspect = width / Math.max(height, 1);
    renderer.setSize(width, height, false);
    // Retain the detailed target and relative zoom when the panel resizes.
    halfSpan = selectedPerson ? halfSpan * Math.max(1.2, 1.15 / aspect) / Math.max(1.2, 1.15 / oldAspect) : fitSpan(camera.quaternion);
    updateProjection();
    if (transition) {
      // Preserve the remaining transition with the newly fitted viewport.
      transition.fromSpan = halfSpan;
    }
    draw();
  }
  function render(topProgress, revealState = timeline.at(0), nextRouter, enabled = false, paused = false, observation = noObservation) {
    latest = { topProgress, state: revealState, nextRouter, enabled, observation };
    setReveal(revealState);
    const posedBody = observation.posed && people.get(observation.posed);
    if (posedBody) posedBody.visible = true;
    const now = performance.now();
    const pulseDelta = !paused && pulseLastAt !== null ? Math.max(0, now - pulseLastAt) / 1000 : 0;
    pulseLastAt = revealState.started && !paused ? now : null;
    routerPulses.forEach((pulse, index) => {
      const visible = revealState.started && (observation.active || revealState.regions[index].complete);
      const time = pulse.material.uniforms.signalTime;
      time.value = observation.active ? observation.time : visible && pulse.visible ? time.value + pulseDelta : 0;
      pulse.visible = visible;
    });
    if (!cameraInteractive) {
      introCamera(topProgress);
      if (topProgress >= 1) {
        cameraInteractive = true;
        controls.enabled = true;
        suppressControlDraw = true;
        controls.update();
        suppressControlDraw = false;
        renderer.domElement.dataset.cameraReady = 'true';
        setMode('top');
      }
    }
    draw();
  }
  resize();
  render(0);
  // Prepare model shaders/textures while the map is opaque and selection disabled.
  // The solid cover still hides all interior geometry; people never render early.
  interior.visible = true;
  await renderer.compileAsync(scene, camera);
  renderer.render(scene, camera);
  // Compile skeleton/icon materials without rendering them (some have depthTest=false).
  occupants.group.visible = true;
  for (const body of people.values()) body.visible = true;
  for (const wave of signalWaves) wave.visible = true;
  for (const pulse of routerPulses) pulse.visible = true;
  wallPreview.visible = true;
  await renderer.compileAsync(scene, camera);
  render(0); // Restore the fully masked initial state before enabling selection.
  return {
    render, resize, interior, setView, resetCamera, focusPerson, clearPersonSelection,
    // Hold camera easing while the reset dialog is open, preserving its target.
    pauseCameraTransition: () => { cancelAnimationFrame(cameraFrame); cameraFrame = 0; },
    excludeDialogTime: duration => {
      if (transition) { transition.at += duration; cameraFrame = requestAnimationFrame(animateView); }
    },
    updatePose: (id, joints) => occupants.updatePose(id, joints),
    resetPose: id => occupants.resetPose(id),
  };
}
