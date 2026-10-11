// Leaflet remains responsible for the actual map. During playback, magnify its
// existing pixels from the shared timeline (no independent flyTo/CSS timer).
export function createMap(settings, onSelect, onTileError) {
  const { latitude, longitude } = settings.center;
  const location = [latitude, longitude];
  const container = document.querySelector('#map');
  const map = L.map(container, { attributionControl: false, zoomControl: false })
    .setView(location, settings.zoom);
  const zoomControl = L.control.zoom({ position: 'topright' }).addTo(map);
  const tiles = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 19 }).addTo(map);
  tiles.on('tileerror', onTileError);

  const marker = L.marker(location, {
    keyboard: false,
    icon: L.divIcon({
      className: 'venue-marker', iconSize: [0, 0], iconAnchor: [0, 0],
      html: '<span class="venue-halo"></span><span class="venue-point"></span><button type="button" class="venue-button" disabled>시연 건물<span>모델 준비 중…</span></button>',
    }),
  }).addTo(map);
  const element = marker.getElement();
  const button = element.querySelector('button');
  const halo = element.querySelector('.venue-halo');
  let locked = false;
  L.DomEvent.disableClickPropagation(button);
  button.addEventListener('click', onSelect);
  marker.on('click', onSelect);

  return {
    setReady() {
      button.disabled = false;
      button.querySelector('span').textContent = '클릭하여 공간 보기';
    },
    lock() {
      locked = true;
      button.disabled = true;
      button.querySelector('span').textContent = '선택 완료';
      map.stop();
      for (const name of ['dragging', 'scrollWheelZoom', 'doubleClickZoom', 'touchZoom', 'boxZoom', 'keyboard']) map[name]?.disable();
      zoomControl.remove();
      document.querySelector('#map-layer').classList.add('is-playing');
    },
    reset() {
      map.stop();
      container.style.transform = '';
      halo.style.opacity = '0';
      halo.style.transform = '';
      document.querySelector('#map-layer').classList.remove('is-playing');
      for (const name of ['dragging', 'scrollWheelZoom', 'doubleClickZoom', 'touchZoom', 'boxZoom', 'keyboard']) map[name]?.enable();
      if (locked) zoomControl.addTo(map);
      locked = false;
      map.setView(location, settings.zoom, { animate: false });
      button.disabled = false;
      button.querySelector('span').textContent = '클릭하여 공간 보기';
    },
    render(focus, selected) {
      if (!selected) return;
      const point = map.latLngToContainerPoint(location);
      const size = map.getSize();
      const scale = 1 + .8 * focus;
      const x = (size.x / 2 - point.x) * focus * scale;
      const y = (size.y / 2 - point.y) * focus * scale;
      container.style.transform = `translate(${x}px, ${y}px) scale(${scale})`;
      halo.style.opacity = String(.45 + .55 * focus);
      halo.style.transform = `scale(${1 + .35 * focus})`;
    },
    resize() { map.invalidateSize({ pan: false, animate: false }); },
  };
}
