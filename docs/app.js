const LABELS = ["intro", "verse", "chorus", "bridge", "inst", "outro", "silence"];
const LABEL_NAMES = {
  intro: "Intro",
  verse: "Verse",
  chorus: "Chorus",
  bridge: "Bridge",
  inst: "Instrumental",
  outro: "Outro",
  silence: "Silence",
};

const state = {
  songs: [],
  current: null,
};

const AUDIO_BASE_URL = "https://huggingface.co/datasets/ASLP-lab/SongFormBench/resolve/main/data/audios/HarmonixSet/";

const elements = {
  list: document.querySelector("#song-list"),
  select: document.querySelector("#song-select"),
  id: document.querySelector("#track-id"),
  title: document.querySelector("#track-title"),
  artist: document.querySelector("#track-artist"),
  duration: document.querySelector("#duration"),
  metricHr05: document.querySelector("#metric-hr05"),
  metricHr3: document.querySelector("#metric-hr3"),
  metricAcc: document.querySelector("#metric-acc"),
  reference: document.querySelector("#reference-track"),
  prediction: document.querySelector("#prediction-track"),
  ruler: document.querySelector("#time-ruler"),
  legend: document.querySelector("#legend"),
  currentTime: document.querySelector("#current-time"),
  tooltip: document.querySelector("#tooltip"),
  sourceLink: document.querySelector("#source-link"),
  audio: document.querySelector("#audio-player"),
};

function formatTime(seconds) {
  const safe = Math.max(0, Number(seconds) || 0);
  const minutes = Math.floor(safe / 60);
  const remainder = Math.floor(safe % 60);
  return `${String(minutes).padStart(2, "0")}:${String(remainder).padStart(2, "0")}`;
}

function formatPrecise(seconds) {
  const safe = Math.max(0, Number(seconds) || 0);
  const minutes = Math.floor(safe / 60);
  const remainder = safe % 60;
  return `${minutes}:${remainder.toFixed(1).padStart(4, "0")}`;
}

function percent(value, duration) {
  return `${Math.max(0, Math.min(100, (value / duration) * 100))}%`;
}

function renderLegend() {
  elements.legend.innerHTML = LABELS.map((label) => `
    <span class="legend-item">
      <i class="legend-swatch label-${label}"></i>${LABEL_NAMES[label]}
    </span>
  `).join("");
}

function showTooltip(event, segment) {
  elements.tooltip.hidden = false;
  elements.tooltip.innerHTML = `<strong>${LABEL_NAMES[segment.label] || segment.label}</strong><br>${formatPrecise(segment.start)}–${formatPrecise(segment.end)}`;
  const margin = 14;
  const width = elements.tooltip.offsetWidth;
  const height = elements.tooltip.offsetHeight;
  elements.tooltip.style.left = `${Math.min(window.innerWidth - width - margin, event.clientX + 12)}px`;
  elements.tooltip.style.top = `${Math.max(margin, event.clientY - height - 10)}px`;
}

function hideTooltip() {
  elements.tooltip.hidden = true;
}

function seekTo(seconds) {
  if (!elements.audio.src) return;
  elements.audio.currentTime = seconds;
  elements.audio.play().catch(() => {});
}

function renderTrack(container, segments, duration) {
  container.innerHTML = "";
  segments.forEach((segment) => {
    const width = ((segment.end - segment.start) / duration) * 100;
    const button = document.createElement("button");
    button.type = "button";
    button.className = `segment label-${segment.label}`;
    button.style.left = percent(segment.start, duration);
    button.style.width = `${Math.max(width, 0.18)}%`;
    button.dataset.narrow = width < 7 ? "true" : "false";
    button.setAttribute("aria-label", `${LABEL_NAMES[segment.label] || segment.label}, ${formatPrecise(segment.start)} to ${formatPrecise(segment.end)}. Seek to section.`);
    button.innerHTML = `<span class="segment-label">${LABEL_NAMES[segment.label] || segment.label}</span>`;
    button.addEventListener("click", () => seekTo(segment.start));
    button.addEventListener("pointerenter", (event) => showTooltip(event, segment));
    button.addEventListener("pointermove", (event) => showTooltip(event, segment));
    button.addEventListener("pointerleave", hideTooltip);
    button.addEventListener("focus", (event) => {
      const rect = event.currentTarget.getBoundingClientRect();
      showTooltip({ clientX: rect.left + rect.width / 2, clientY: rect.top }, segment);
    });
    button.addEventListener("blur", hideTooltip);
    container.appendChild(button);
  });

  const playhead = document.createElement("div");
  playhead.className = "playhead";
  playhead.style.left = "0%";
  playhead.setAttribute("aria-hidden", "true");
  container.appendChild(playhead);
}

function renderRuler(duration) {
  const ticks = [0, 0.25, 0.5, 0.75, 1];
  elements.ruler.innerHTML = ticks.map((ratio) => `
    <span class="tick" style="left:${ratio * 100}%">${formatTime(duration * ratio)}</span>
  `).join("");
}

function renderSongList() {
  elements.list.innerHTML = "";
  elements.select.innerHTML = "";

  state.songs.forEach((song) => {
    const item = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.className = "song-button";
    button.dataset.id = song.id;
    button.innerHTML = `<span><span class="song-title">${song.title}</span><span class="song-artist">${song.artist}</span></span>`;
    button.addEventListener("click", () => selectSong(song.id));
    item.appendChild(button);
    elements.list.appendChild(item);

    const option = document.createElement("option");
    option.value = song.id;
    option.textContent = `${song.title} — ${song.artist}`;
    elements.select.appendChild(option);
  });

  elements.select.addEventListener("change", () => selectSong(elements.select.value));
}

function getAudioUrl(song) {
  return `${AUDIO_BASE_URL}${encodeURIComponent(song.id)}.wav`;
}

function renderSong(song) {
  state.current = song;
  elements.id.textContent = song.id;
  elements.title.textContent = song.title;
  elements.artist.textContent = song.artist;
  elements.duration.textContent = formatTime(song.duration) + " source duration";
  const audioUrl = getAudioUrl(song);
  elements.sourceLink.href = audioUrl;
  elements.metricHr05.textContent = song.metrics.hr05f.toFixed(3);
  elements.metricHr3.textContent = song.metrics.hr3f.toFixed(3);
  elements.metricAcc.textContent = song.metrics.acc.toFixed(3);
  elements.select.value = song.id;

  document.querySelectorAll(".song-button").forEach((button) => {
    button.setAttribute("aria-current", button.dataset.id === song.id ? "true" : "false");
  });

  renderRuler(song.duration);
  renderTrack(elements.reference, song.reference, song.duration);
  renderTrack(elements.prediction, song.prediction, song.duration);
  updatePlayhead(0);

  elements.audio.pause();
  elements.audio.src = audioUrl;
  elements.audio.load();

  const url = new URL(window.location.href);
  url.searchParams.set("song", song.id);
  window.history.replaceState({}, "", url);
}

function selectSong(id) {
  const song = state.songs.find((item) => item.id === id) || state.songs[0];
  if (!song || state.current?.id === song.id) return;
  renderSong(song);
}

function updatePlayhead(seconds) {
  if (!state.current) return;
  const ratio = Math.max(0, Math.min(1, seconds / state.current.duration));
  document.querySelectorAll(".playhead").forEach((node) => {
    node.style.left = `${ratio * 100}%`;
  });
  elements.currentTime.textContent = `${formatTime(seconds)} / ${formatTime(state.current.duration)}`;
}

elements.audio.addEventListener("timeupdate", () => updatePlayhead(elements.audio.currentTime));
elements.audio.addEventListener("seeked", () => updatePlayhead(elements.audio.currentTime));
elements.audio.addEventListener("ended", () => updatePlayhead(0));
elements.audio.addEventListener("error", () => {
  if (!state.current) return;
  elements.currentTime.textContent = "Audio unavailable — open the WAV source";
});

async function init() {
  renderLegend();
  const response = await fetch("./data/songs.json");
  if (!response.ok) throw new Error("Unable to load demo data.");
  state.songs = await response.json();
  renderSongList();

  const requested = new URL(window.location.href).searchParams.get("song");
  const initial = state.songs.find((song) => song.id === requested) || state.songs[0];
  renderSong(initial);
}

init().catch((error) => {
  elements.title.textContent = "Demo data could not be loaded";
  elements.artist.textContent = error.message;
});
