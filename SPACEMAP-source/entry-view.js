import * as THREE from 'three';
import { Line2 } from 'three/addons/lines/Line2.js';
import { LineGeometry } from 'three/addons/lines/LineGeometry.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';

// This marker follows the prepared route. It is not one of the four people.
export function createEntryView({ scene, camera, container, layout, onRouterClick }) {
  const green = 0x74d682;
  // Existing GLB floor surfaces (metres), not the A–D reveal rectangles.
  // The entrance and west bedroom are .05 m below the surrounding corridor.
  const lowerFloors = [
    { x: [11.02, 13.44], z: [-7.439, -5.39] },
    { x: [2.57, 4.19], z: [-7.71, -5.39] },
  ];
  const surfaceY = (x, z) => (lowerFloors.some(floor =>
    x >= floor.x[0] && x <= floor.x[1] && z >= floor.z[0] && z <= floor.z[1]
  ) ? .20 : .25) + .025;

  function floorRoute(points) {
    const vertices = [];
    for (let i = 0; i < points.length - 1; i++) {
      const [x, z] = points[i], [nx, nz] = points[i + 1];
      const dx = nx - x, dz = nz - z;
      const cuts = [0, 1];
      for (const floor of lowerFloors) {
        for (const edge of floor.x) {
          const t = (edge - x) / dx;
          if (t > 0 && t < 1 && z + dz * t >= floor.z[0] && z + dz * t <= floor.z[1]) cuts.push(t);
        }
        for (const edge of floor.z) {
          const t = (edge - z) / dz;
          if (t > 0 && t < 1 && x + dx * t >= floor.x[0] && x + dx * t <= floor.x[1]) cuts.push(t);
        }
      }
      const sorted = [...new Set(cuts)].sort((a, b) => a - b);
      for (let j = 0; j < sorted.length - 1; j++) {
        const from = sorted[j], to = sorted[j + 1], mid = (from + to) / 2;
        const y = surfaceY(x + dx * mid, z + dz * mid);
        // Split at each floor step so long corridor segments do not slope.
        vertices.push([x + dx * from, y, z + dz * from], [x + dx * to, y, z + dz * to]);
      }
    }
    return vertices;
  }
  const marker = new THREE.Group();
  marker.name = 'Demo entrant';
  marker.visible = false;
  scene.add(marker);

  function markerPart(geometry, color, order) {
    const mesh = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({
      color, side: THREE.DoubleSide, transparent: true, depthTest: false, depthWrite: false, toneMapped: false,
    }));
    mesh.rotation.x = -Math.PI / 2;
    mesh.renderOrder = order;
    marker.add(mesh);
    return mesh;
  }
  markerPart(new THREE.CircleGeometry(.21, 40), 0x0b160f, 50);
  markerPart(new THREE.RingGeometry(.185, .21, 40), green, 51);
  const arrow = new THREE.Shape();
  arrow.moveTo(0, .145);
  arrow.lineTo(-.105, -.015);
  arrow.lineTo(-.035, .005);
  arrow.lineTo(-.035, -.125);
  arrow.lineTo(.035, -.125);
  arrow.lineTo(.035, .005);
  arrow.lineTo(.105, -.015);
  arrow.closePath();
  markerPart(new THREE.ShapeGeometry(arrow), green, 52);

  let trailGeometry = new LineGeometry();
  const trailMaterial = new LineMaterial({
    color: green, linewidth: 1.8, dashed: true, dashSize: .17, gapSize: .13,
    transparent: true, opacity: .8, depthTest: false, depthWrite: false, toneMapped: false,
  });
  const trail = new Line2(trailGeometry, trailMaterial);
  trail.name = 'Traveled demo route';
  trail.renderOrder = 45;
  trail.visible = false;
  trail.frustumCulled = false;
  scene.add(trail);
  let lastTrail = '';

  const pins = document.createElement('div');
  pins.className = 'router-pins';
  pins.hidden = true;
  pins.setAttribute('role', 'group');
  pins.setAttribute('aria-label', '사전 배치된 가상 라우터');
  container.appendChild(pins);
  const buttons = new Map(layout.routers.map(router => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'router-pin';
    button.textContent = router.id;
    button.disabled = true;
    let pointerStart = null, dragged = false;
    button.addEventListener('pointerdown', event => {
      event.stopPropagation();
      pointerStart = [event.clientX, event.clientY];
      dragged = false;
      button.setPointerCapture(event.pointerId);
    });
    button.addEventListener('pointermove', event => {
      event.stopPropagation();
      if (pointerStart && Math.hypot(event.clientX - pointerStart[0], event.clientY - pointerStart[1]) > 5) dragged = true;
    });
    button.addEventListener('pointerup', event => { event.stopPropagation(); pointerStart = null; });
    button.addEventListener('pointercancel', () => { pointerStart = null; dragged = true; });
    button.addEventListener('contextmenu', event => event.preventDefault());
    button.addEventListener('click', event => {
      event.stopPropagation();
      if (!button.disabled && (event.detail === 0 || !dragged)) onRouterClick?.(router.id);
    });
    pins.appendChild(button);
    return [router.id, button];
  }));
  const projected = new THREE.Vector3();
  const cameraSpace = new THREE.Vector3();
  let width = 1, height = 1;

  function resize() {
    width = Math.max(container.clientWidth, 1);
    height = Math.max(container.clientHeight, 1);
    trailMaterial.resolution.set(width, height);
  }

  function render(state, nextRouter, enabled = false) {
    resize();
    pins.hidden = !state.started;
    marker.visible = Boolean(state.started && state.entry);
    const points = state.entry?.trail ?? [];
    trail.visible = Boolean(state.started && points.length > 1);
    if (marker.visible) {
      const [x, z] = state.entry.position;
      const [dx, dz] = state.entry.direction;
      marker.position.set(x, surfaceY(x, z), z);
      if (dx || dz) marker.rotation.y = Math.atan2(-dx, -dz);
    }
    // Completed route segments stay in the line; only the moving tip changes.
    const key = JSON.stringify(points);
    if (trail.visible && key !== lastTrail) {
      // A growing instanced line needs fresh geometry: Three caches the initial
      // segment capacity. Dispose the old GPU buffers before replacing it.
      trailGeometry.dispose();
      trailGeometry = new LineGeometry();
      trail.geometry = trailGeometry;
      const vertices = floorRoute(points);
      trailGeometry.setPositions(vertices.flat());
      trail.computeLineDistances();
      // Anchor dashes to each segment's world direction. Walking back along
      // the same corridor then keeps the same gaps instead of filling them in.
      const starts = trailGeometry.attributes.instanceDistanceStart;
      const ends = trailGeometry.attributes.instanceDistanceEnd;
      for (let i = 0; i < vertices.length - 1; i++) {
        const [x, , z] = vertices[i], [nx, , nz] = vertices[i + 1];
        const length = Math.hypot(nx - x, nz - z) || 1;
        const sign = nx < x || (nx === x && nz < z) ? -1 : 1;
        const dx = (nx - x) / length * sign, dz = (nz - z) / length * sign;
        starts.setX(i, x * dx + z * dz);
        ends.setX(i, nx * dx + nz * dz);
      }
      starts.data.needsUpdate = true;
      lastTrail = key;
    }
    if (!state.started) lastTrail = '';

    camera.updateMatrixWorld();
    for (const router of layout.routers) {
      const button = buttons.get(router.id);
      const region = state.regions.find(item => item.router === router.id);
      const complete = Boolean(region?.complete);
      const active = !complete && (region?.progress ?? 0) > 0;
      button.disabled = !(state.started && enabled && nextRouter === router.id);
      button.dataset.state = complete ? 'complete' : active ? 'active' : 'waiting';
      // Inline glow uses the same reveal fraction and ends with that step.
      const glow = active ? Math.sin(Math.PI * region.progress) : 0;
      button.style.boxShadow = glow > 0 ? `0 0 ${8 + glow * 10}px rgba(116, 214, 130, ${glow * .3})` : 'none';
      const description = complete ? '공개 완료' : active ? '가상 신호 수신 중' : button.disabled ? '사전 배치된 가상 라우터' : '클릭하여 가상 신호 수신';
      button.title = `${router.id} · ${description}`;
      button.setAttribute('aria-label', `${router.id} ${description}`);
      // Labels follow the real device floor height as the camera moves.
      projected.set(router.position[0], router.position[1] + .35, router.position[2]);
      cameraSpace.copy(projected).applyMatrix4(camera.matrixWorldInverse);
      projected.project(camera);
      button.hidden = cameraSpace.z >= 0 || Math.abs(projected.x) > 1 || Math.abs(projected.y) > 1 || Math.abs(projected.z) > 1;
      button.style.left = `${(projected.x * .5 + .5) * width}px`;
      button.style.top = `${(-projected.y * .5 + .5) * height}px`;
    }
  }
  resize();
  return { render, resize };
}
