// SignTalk phone client.
// Camera -> MediaPipe (hands + pose, in this browser) -> landmark window -> Pi /api/predict -> speak.
// Only landmark numbers leave the phone, never video.
'use strict';

const $ = (s) => document.querySelector(s);
const POSE_IDX = [0, 11, 12, 13, 14, 15, 16];                   // nose, shoulders, elbows, wrists (same as training)
const HAND_LINKS = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17]];
const ARM_LINKS = [[11, 12], [11, 13], [13, 15], [12, 14], [14, 16]];
const REC_SECONDS = 3.0;                                          // training clips were 3 s windows
const RES = { 360: [640, 360], 720: [1280, 720], 1080: [1920, 1080] };

const store = {
  get(k, d) { try { const v = localStorage.getItem('st.' + k); return v === null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem('st.' + k, JSON.stringify(v)); } catch { /* private mode */ } },
};
const S = {
  cfg: null, stream: null, hands: null, pose: null, ready: false, mode: 'idle', buf: [], lastTs: 0, ar: 16 / 9,
  facing: store.get('cam', 'environment'), res: store.get('res', 720), cd: store.get('cd', 2),
  speakOn: store.get('speak', true), vol: store.get('vol', 1), voiceURI: store.get('voice', ''), beep: store.get('beep', true),
  gpu: new URLSearchParams(location.search).get('gpu') === '0' ? false : store.get('gpu', true), fps: 0, lastFrame: 0, handsN: 0, bodyOk: false, lastBody: 0, lastHands: 0, voice: null,
  history: [], audio: null, wake: null,
  ring: [], up: false, tab: 'translate', paused: false, _live: null,
  autoOn: store.get('auto', true), thr: store.get('thr', null), raiseThr: store.get('raise', 0.8),
  auto: { state: 'rest', runStart: 0, tStart: 0, downSince: 0, coolUntil: 0 },
};
const video = $('#video'), canvas = $('#overlay'), ctx = canvas.getContext('2d'), vp = $('#vp');

function rlog(msg) {
  try { fetch('/api/log', { method: 'POST', body: JSON.stringify({ msg: String(msg) }), keepalive: true }).catch(() => {}); } catch { /* ignore */ }
}
window.addEventListener('error', (e) => rlog(`js error: ${e.message} @ ${e.filename}:${e.lineno}`));
window.addEventListener('unhandledrejection', (e) => rlog(`promise rejected: ${e.reason}`));

let toastTimer = 0;
function toast(msg, ms = 2600) {
  const t = $('#toast'); t.textContent = msg; t.classList.add('on');
  clearTimeout(toastTimer); toastTimer = setTimeout(() => t.classList.remove('on'), ms);
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function api(path, body) {
  const r = await fetch('/api/' + path, body === undefined ? {} : {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  });
  return r.json();
}
function loadMsg(txt, err = false) {
  $('#loadTxt').textContent = txt; $('#loading').classList.toggle('err', err); $('#loading').classList.remove('off');
}

/* ---------- layout ---------- */
function fit() {
  const st = document.querySelector('.stage');
  const wide = innerWidth >= 900 || (innerHeight <= 520 && innerWidth > innerHeight);
  const maxH = wide ? innerHeight - 120 : innerHeight * 0.5;
  const w = Math.max(160, Math.min(st.clientWidth, maxH * S.ar)), h = w / S.ar;
  vp.style.width = w + 'px'; vp.style.height = h + 'px';
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
}
addEventListener('resize', fit);

/* ---------- camera ---------- */
async function startCamera() {
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    throw new Error('The camera needs a secure page. Open the https:// address of the Pi (accept the certificate warning once).');
  }
  if (S.stream) S.stream.getTracks().forEach((t) => t.stop());
  const [w, h] = RES[S.res] || RES[720];
  try {
    S.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: S.facing }, width: { ideal: w }, height: { ideal: h }, frameRate: { ideal: 30, max: 30 } }, audio: false,
    });
  } catch (e) {
    if (e.name === 'NotAllowedError') throw new Error('Camera permission was denied. Allow the camera for this site in the browser, then reload.');
    if (e.name === 'NotFoundError') throw new Error('No camera found on this device.');
    throw new Error('Could not start the camera: ' + e.message);
  }
  video.srcObject = S.stream;
  await video.play();
  if (!video.videoWidth) await new Promise((r) => video.addEventListener('loadedmetadata', r, { once: true }));
  S.ar = video.videoWidth / video.videoHeight;
  vp.classList.toggle('mirror', S.facing === 'user');
  fit();
  const t = S.stream.getVideoTracks()[0].getSettings();
  rlog(`camera ${t.width}x${t.height}@${Math.round(t.frameRate || 0)} facing=${t.facingMode || S.facing} ua=${navigator.userAgent}`);
}

/* ---------- MediaPipe ---------- */
async function loadModels() {
  loadMsg('Loading hand tracker…');
  const { FilesetResolver, HandLandmarker, PoseLandmarker } = await import('/mp/vision_bundle.mjs');
  const fileset = await FilesetResolver.forVisionTasks('/mp/wasm');
  const make = async (Cls, file, extra) => {
    const opts = (delegate) => ({ baseOptions: { modelAssetPath: '/mp/' + file, delegate }, runningMode: 'VIDEO', ...extra });
    if (S.gpu) { try { return [await Cls.createFromOptions(fileset, opts('GPU')), 'GPU']; } catch (e) { rlog('GPU delegate failed: ' + e); } }
    return [await Cls.createFromOptions(fileset, opts('CPU')), 'CPU'];
  };
  let dh, dp;
  [S.hands, dh] = await make(HandLandmarker, 'hand_landmarker.task', { numHands: 2 });
  loadMsg('Loading body tracker…');
  [S.pose, dp] = await make(PoseLandmarker, 'pose_landmarker_lite.task', {});
  S.delegate = `${dh}/${dp}`;
  rlog('models loaded: ' + S.delegate);
}

const r4 = (v) => Math.round(v * 10000) / 10000;
function pack(h, p) {
  let pose = null;
  const pl = p.landmarks && p.landmarks[0];
  if (pl) { pose = []; for (const i of POSE_IDX) pose.push(r4(pl[i].x), r4(pl[i].y)); }
  let lh = null, rh = null, sl = -1, sr = -1;
  (h.landmarks || []).forEach((l, i) => {
    const c = h.handedness[i][0];
    const flat = []; for (const q of l) flat.push(r4(q.x), r4(q.y));
    if (c.categoryName === 'Left' && c.score > sl) { lh = flat; sl = c.score; }
    if (c.categoryName === 'Right' && c.score > sr) { rh = flat; sr = c.score; }
  });
  return [pose, lh, rh];
}

function draw(h, p) {
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  const lw = Math.max(2, W / 320);
  ctx.lineCap = 'round'; ctx.lineJoin = 'round';
  const pl = p.landmarks && p.landmarks[0];
  if (pl) {
    ctx.strokeStyle = 'rgba(233,237,255,.55)'; ctx.lineWidth = lw * 1.2;
    ctx.beginPath(); for (const [a, b] of ARM_LINKS) { ctx.moveTo(pl[a].x * W, pl[a].y * H); ctx.lineTo(pl[b].x * W, pl[b].y * H); } ctx.stroke();
    ctx.fillStyle = 'rgba(233,237,255,.9)';
    for (const i of POSE_IDX) { ctx.beginPath(); ctx.arc(pl[i].x * W, pl[i].y * H, lw * 1.6, 0, 7); ctx.fill(); }
  }
  (h.landmarks || []).forEach((l, i) => {
    const left = h.handedness[i][0].categoryName === 'Left';
    const col = left ? '61,220,151' : '255,180,84';
    ctx.strokeStyle = `rgba(${col},1)`; ctx.lineWidth = lw * 1.3;
    ctx.beginPath(); for (const [a, b] of HAND_LINKS) { ctx.moveTo(l[a].x * W, l[a].y * H); ctx.lineTo(l[b].x * W, l[b].y * H); } ctx.stroke();
    ctx.shadowBlur = 0; ctx.fillStyle = '#fff';
    for (const q of l) { ctx.beginPath(); ctx.arc(q.x * W, q.y * H, lw * 1.25, 0, 7); ctx.fill(); }
  });
}

/* ---------- per-frame loop ---------- */
function step(now) {
  if (!S.ready || video.readyState < 2) return;
  const ts = Math.max(S.lastTs + 1, Math.round(now));
  S.lastTs = ts;
  let h, p;
  try { h = S.hands.detectForVideo(video, ts); p = S.pose.detectForVideo(video, ts); }
  catch (e) { if (!S.errLogged) { S.errLogged = true; rlog('detect error: ' + e); } return; }
  draw(h, p);
  const fr = pack(h, p);
  if (S.mode === 'rec' && S.buf.length < 600) S.buf.push(fr);
  const t = performance.now();
  const up = isRaised(h, p.landmarks && p.landmarks[0]);
  S.ring.push({ t, fr, up });
  while (S.ring.length && t - S.ring[0].t > 6000) S.ring.shift();
  S.up = up;
  autoStep(t, up);
  if (S.lastFrame) { const dt = t - S.lastFrame; S.fps = S.fps ? S.fps * 0.9 + (1000 / dt) * 0.1 : 1000 / dt; }
  S.lastFrame = t;
  S.handsN = (h.landmarks || []).length;
  const pl = p.landmarks && p.landmarks[0];
  S.bodyOk = !!pl && [11, 12].every((i) => pl[i].x > 0.02 && pl[i].x < 0.98 && pl[i].y > 0.02 && pl[i].y < 0.98);
  if (S.bodyOk) S.lastBody = t;
  if (S.handsN) S.lastHands = t;
}
function startLoop() {
  if ('requestVideoFrameCallback' in video) {
    const cb = (now) => { step(now); video.requestVideoFrameCallback(cb); };
    video.requestVideoFrameCallback(cb);
  } else {
    const raf = (now) => { step(now); requestAnimationFrame(raf); };
    requestAnimationFrame(raf);
  }
}
function updatePills() {
  const t = performance.now();
  const body = $('#pBody'), hands = $('#pHands');
  body.classList.toggle('ok', S.bodyOk); body.classList.toggle('bad', !S.bodyOk && S.ready);
  hands.classList.toggle('ok', S.handsN > 0); hands.querySelector('em').textContent = 'Hands ' + S.handsN;
  $('#pFps em').textContent = Math.round(S.fps) + ' fps';
  let hint = '';
  if (S.ready && S.mode === 'idle') {
    if (t - S.lastBody > 1500) hint = 'Move back so your head and shoulders are in view';
    else if (t - S.lastHands > 2500) hint = 'Raise your hands into view';
  }
  const el = $('#hint'); el.textContent = hint; el.classList.toggle('on', !!hint);
}
setInterval(updatePills, 300);

async function ping() {
  try {
    const j = await api('ping');
    const el = $('#pPi'); el.classList.add('ok'); el.classList.remove('bad');
    el.querySelector('em').textContent = 'Pi' + (j.temp ? ` ${Math.round(j.temp)}°C` : '');
    S.piTemp = j.temp;
  } catch {
    const el = $('#pPi'); el.classList.remove('ok'); el.classList.add('bad'); el.querySelector('em').textContent = 'Pi offline';
  }
}
setInterval(ping, 5000);

/* ---------- hands-free mode ----------
   A sign starts when a hand is lifted above the lap (wrist clearly above the shoulder line + RAISE x shoulder width).
   The Pi gets the window [start - 1 s, start + 2 s], the same layout as the training clips (rest, then the sign).
   After an answer the hands must go down again before the next sign is accepted. */
function isRaised(h, pl) {
  if (!pl || !h.landmarks || !h.landmarks.length) return false;
  const sw = Math.abs(pl[11].x - pl[12].x) * S.ar;                 // shoulder width in "image-height" units
  if (sw < 0.02) return false;
  const sy = (pl[11].y + pl[12].y) / 2;
  return h.landmarks.some((l) => (l[0].y - sy) / sw < S.raiseThr);
}
const LIVE = {
  listen: ['Listening… lift your hands and sign', 'listen'], sign: ['Sign detected, keep signing', 'sign'],
  think: ['Thinking…', 'think'], rest: ['Lower your hands to continue', 'rest'], paused: ['Paused', ''], off: ['', ''],
};
function setLive(k) {
  if (S._live === k) return;
  S._live = k;
  const el = $('#live'); el.hidden = !S.autoOn; el.className = 'live ' + LIVE[k][1]; $('#liveTxt').textContent = LIVE[k][0];
}
function autoActive() { return S.autoOn && !S.paused && S.ready && S.mode === 'idle' && S.tab === 'translate'; }
function autoStep(t, up) {
  const A = S.auto;
  if (!autoActive()) {
    if (A.state !== 'rest') { A.state = 'rest'; A.downSince = 0; $('#progress').classList.remove('on'); }
    setLive(S.autoOn && S.paused ? 'paused' : 'off');
    return;
  }
  switch (A.state) {
    case 'arm':
      setLive('listen');
      if (up) {
        A.runStart = A.runStart || t;
        if (t - A.runStart >= 250) { A.state = 'sign'; A.tStart = A.runStart; beep(1040, 60); setLive('sign'); rlog('auto: sign detected'); }
      } else A.runStart = 0;
      break;
    case 'sign': {
      const e = t - A.tStart;
      $('#progress').classList.add('on'); $('#progress b').style.width = Math.min(100, e / 20) + '%';
      if (e >= 2000) {
        A.state = 'think'; $('#progress').classList.remove('on'); setLive('think');
        sendWindow(A.tStart - 1000, A.tStart + 2000);
      }
      break;
    }
    case 'rest':
      setLive('rest');
      if (!up && t >= A.coolUntil) {
        A.downSince = A.downSince || t;
        if (t - A.downSince >= 600) { A.state = 'arm'; A.runStart = 0; A.downSince = 0; }
      } else A.downSince = 0;
      break;
    default: break;                                                    // 'think': waiting for the Pi
  }
}
async function sendWindow(a, b) {
  const A = S.auto;
  const win = S.ring.filter((r) => r.t >= a && r.t <= b);
  const upFrac = win.length ? win.filter((r) => r.up).length / win.length : 0;
  const finish = (cool) => { A.state = 'rest'; A.downSince = 0; A.coolUntil = performance.now() + cool; };
  if (win.length < 12 || upFrac < 0.25) { rlog(`auto: ignored (frames ${win.length}, hand up ${Math.round(upFrac * 100)}%)`); finish(300); return; }
  try {
    const j = await api('predict', { aspect: S.ar, thr: S.thr, frames: win.map((r) => r.fr) });
    showResult(j);
    rlog(`auto: ${j.best ? j.best.id + ' ' + j.best.p : j.reason} ${j.accepted ? 'spoken' : 'not spoken'} (frames ${win.length})`);
  } catch (e) { showResult({ error: 'Could not reach the Pi: ' + e.message }); }
  finish(1200);
}
function applyMode() {
  S._live = null;
  document.querySelectorAll('#modeSeg button').forEach((b) => b.classList.toggle('on', (b.dataset.m === 'auto') === S.autoOn));
  $('#live').hidden = !S.autoOn;
  $('#ctaTxt').textContent = S.autoOn ? (S.paused ? 'Resume listening' : 'Pause listening') : 'Tap to sign';
  $('#btnSign').disabled = !S.ready;
  S.auto.state = 'rest'; S.auto.downSince = 0;
  const card = $('#result');
  if (card.dataset.state === 'idle') {
    $('#phrase').textContent = S.autoOn ? 'Just sign' : 'Tap, then sign';
    $('#msg').textContent = S.autoOn ? 'Lift your hands from your lap and sign one phrase. The answer is spoken automatically.'
      : 'Press the button, wait for the countdown and sign one phrase.';
  }
}
function togglePause() { S.paused = !S.paused; applyMode(); }

/* ---------- capture (countdown + 3 s window) ---------- */
function beep(freq = 880, ms = 90) {
  if (!S.beep) return;
  try {
    S.audio = S.audio || new (window.AudioContext || window.webkitAudioContext)();
    const o = S.audio.createOscillator(), g = S.audio.createGain();
    o.frequency.value = freq; g.gain.value = 0.08 * (S.vol || 1);
    o.connect(g).connect(S.audio.destination); o.start(); o.stop(S.audio.currentTime + ms / 1000);
  } catch { /* no audio */ }
}
function bigText(txt, go = false) {
  const b = $('#big'); b.textContent = txt; b.className = 'big'; void b.offsetWidth; b.className = 'big pop' + (go ? ' go' : '');
}
async function capture(cd) {
  S.mode = 'countdown';
  for (let n = cd; n > 0; n--) { bigText(n); if (n <= 3) beep(520, 60); await sleep(1000); }
  bigText('SIGN!', true); beep(1040, 140); if (navigator.vibrate) navigator.vibrate(40);
  S.buf = []; S.mode = 'rec';
  $('#rec').classList.add('on'); $('#progress').classList.add('on');
  const bar = $('#progress b'), t0 = performance.now();
  await new Promise((res) => {
    const tick = () => {
      const e = (performance.now() - t0) / 1000;
      bar.style.width = Math.min(100, (e / REC_SECONDS) * 100) + '%';
      $('#recTxt').textContent = 'REC ' + Math.max(0, REC_SECONDS - e).toFixed(1) + 's';
      e >= REC_SECONDS ? res() : requestAnimationFrame(tick);
    };
    tick();
  });
  $('#rec').classList.remove('on'); $('#progress').classList.remove('on'); bar.style.width = '0';
  const frames = S.buf; S.buf = []; S.mode = 'sending';
  return frames;
}

/* ---------- speech ---------- */
function pickVoice() {
  if (!('speechSynthesis' in window)) return;
  const vs = speechSynthesis.getVoices();
  const sel = $('#setVoice');
  if (vs.length && sel.options.length !== vs.length + 1) {
    sel.innerHTML = '<option value="">Automatic (English)</option>' + vs.map((v) => `<option value="${v.voiceURI}">${v.name} (${v.lang})</option>`).join('');
    sel.value = S.voiceURI;
  }
  S.voice = vs.find((v) => v.voiceURI === S.voiceURI)
    || vs.find((v) => /^en[-_]IN/i.test(v.lang)) || vs.find((v) => /^en[-_](GB|US|AU)/i.test(v.lang)) || vs.find((v) => /^en/i.test(v.lang)) || null;
}
if ('speechSynthesis' in window) { speechSynthesis.addEventListener?.('voiceschanged', pickVoice); pickVoice(); }
function speak(text) {
  if (!S.speakOn || !('speechSynthesis' in window)) return;
  try {
    speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    if (S.voice) { u.voice = S.voice; u.lang = S.voice.lang; } else u.lang = 'en-IN';
    u.volume = S.vol; u.rate = 0.95; u.pitch = 1;
    speechSynthesis.speak(u);
  } catch (e) { rlog('speech error: ' + e); }
}
function primeSpeech() {            // iOS/Android only allow speech after a tap
  if (!('speechSynthesis' in window)) return;
  try { const u = new SpeechSynthesisUtterance(' '); u.volume = 0; speechSynthesis.speak(u); } catch { /* ignore */ }
}

/* ---------- translate tab ---------- */
const REASONS = {
  no_hands: ["Couldn't see your hands", 'Keep both hands in the picture while you sign.'],
  no_body: ["Couldn't see you", 'Step back so your head and shoulders are visible.'],
  too_short: ['Too short', 'The phone was too slow to record. Close other tabs and try again.'],
  unsure: ['Not sure', 'Try again, a bit slower and clearer.'],
};
function showResult(j) {
  const card = $('#result');
  card.classList.remove('pulse'); void card.offsetWidth; card.classList.add('pulse');
  $('#alts').innerHTML = '';
  if (j.error) {
    card.dataset.state = 'warn'; $('#kicker').textContent = 'Error'; $('#phrase').textContent = 'Something went wrong';
    $('#msg').textContent = j.error; $('#confBar').style.width = '0'; $('#confTxt').textContent = ''; return;
  }
  const alts = (j.top || []).slice(j.accepted ? 1 : 0).map((t) => `<li>${t.label} ${Math.round(t.p * 100)}%</li>`).join('');
  if (j.accepted) {
    card.dataset.state = 'ok'; $('#kicker').textContent = 'Recognised';
    $('#phrase').textContent = j.best.label; $('#msg').textContent = `${Math.round(j.best.p * 100)}% sure · answered in ${j.ms} ms`;
    $('#confBar').style.width = Math.round(j.best.p * 100) + '%'; $('#confTxt').textContent = Math.round(j.best.p * 100) + '%';
    $('#alts').innerHTML = alts;
    speak(j.best.speak);
    if (navigator.vibrate) navigator.vibrate([30, 40, 30]);
    S.history.unshift(j.best); S.history = S.history.slice(0, 8); renderHistory();
    document.querySelectorAll('#known .chip').forEach((c) => c.classList.toggle('hit', c.dataset.id === j.best.id));
    setTimeout(() => document.querySelectorAll('#known .chip.hit').forEach((c) => c.classList.remove('hit')), 4000);
  } else {
    const [title, msg] = REASONS[j.reason] || REASONS.unsure;
    card.dataset.state = 'warn'; $('#kicker').textContent = 'No answer'; $('#phrase').textContent = title;
    $('#msg').textContent = j.best ? `${msg} Best guess: ${j.best.label} ${Math.round(j.best.p * 100)}%.` : msg;
    $('#confBar').style.width = j.best ? Math.round(j.best.p * 100) + '%' : '0'; $('#confTxt').textContent = j.best ? Math.round(j.best.p * 100) + '%' : '';
    $('#alts').innerHTML = alts;
  }
}
function renderHistory() {
  const h = $('#history');
  h.innerHTML = S.history.length ? '' : '<span class="empty">Nothing yet</span>';
  S.history.forEach((t) => {
    const b = document.createElement('button'); b.className = 'chip'; b.textContent = t.label; b.onclick = () => speak(t.speak); h.appendChild(b);
  });
}
function renderKnown() {
  const k = $('#known'); k.innerHTML = '';
  S.cfg.labels.forEach((l) => {
    const b = document.createElement('button'); b.className = 'chip'; b.dataset.id = l.id; b.textContent = l.label;
    b.title = 'Tap to hear it'; b.onclick = () => { primeSpeech(); speak(l.speak); }; k.appendChild(b);
  });
}
function setBusy(btn, busy, txt) {
  btn.disabled = busy || !S.ready; btn.classList.toggle('busy', busy);
  if (txt && btn.id === 'btnSign') $('#ctaTxt').textContent = txt;
}
async function onSign() {
  if (S.mode !== 'idle') return;
  primeSpeech(); const b = $('#btnSign'); setBusy(b, true, 'Get ready…');
  try {
    const frames = await capture(S.cd);
    setBusy(b, true, 'Thinking…');
    const j = await api('predict', { aspect: S.ar, thr: S.thr, frames });
    showResult(j);
    return j;
  } catch (e) { showResult({ error: 'Could not reach the Pi: ' + e.message }); }
  finally { S.mode = 'idle'; setBusy(b, false, 'Tap to sign'); }
}

/* ---------- teach tab ---------- */
function teachPhrase() {
  const sel = $('#teachSel');
  return sel.value === '__new' ? $('#teachNew').value.trim() : sel.value;
}
function renderTeach() {
  const sel = $('#teachSel'), keep = sel.value;
  const ids = new Map();
  S.cfg.labels.forEach((l) => ids.set(l.id, l.label));
  (S.cfg.new_phrases || []).forEach((l) => ids.set(l.id, l.label + ' (new)'));
  sel.innerHTML = [...ids].map(([id, lab]) => `<option value="${id}">${lab}</option>`).join('') + '<option value="__new">➕ New phrase…</option>';
  if (keep && [...sel.options].some((o) => o.value === keep)) sel.value = keep;
  teachCount();
}
function teachCount() {
  const sel = $('#teachSel'); $('#newWrap').hidden = sel.value !== '__new';
  const c = (S.cfg.clips || {})[sel.value] || { phone: 0, total: 0 };
  $('#teachCount').innerHTML = sel.value === '__new' ? 'Type the phrase as it should be spoken.'
    : `<b>${c.phone}</b> clips from phones · <b>${c.total}</b> in total` + (c.total < 5 ? ' · <span style="color:var(--warn)">need at least 5</span>' : '');
}
async function refreshCfg() {
  S.cfg = await api('config'); renderKnown(); renderTeach();
}
async function onTeach() {
  if (S.mode !== 'idle') return;
  const phrase = teachPhrase();
  if (!phrase) { toast('Choose or type a phrase first'); return; }
  primeSpeech(); const b = $('#btnTeach'); b.disabled = true; b.classList.add('busy');
  try {
    const frames = await capture(S.cd);
    const j = await api('teach', { phrase, aspect: S.ar, frames });
    $('#teachMsg').textContent = j.saved ? `Saved clip ${j.count} for "${j.phrase}" (hands seen ${Math.round(j.hands_seen * 100)}%).` : (j.error || 'Not saved');
    if (j.saved) { navigator.vibrate?.(30); await refreshCfg(); $('#teachSel').value = j.phrase; teachCount(); }
  } catch (e) { $('#teachMsg').textContent = 'Could not reach the Pi: ' + e.message; }
  finally { S.mode = 'idle'; b.disabled = !S.ready; b.classList.remove('busy'); S.auto.state = 'rest'; S.auto.downSince = 0; }
}
let trainTimer = 0;
async function pollTrain() {
  const j = await api('retrain_status');
  const bar = $('#trainBar'); bar.hidden = j.state !== 'running' && j.state !== 'done';
  bar.firstElementChild.style.width = Math.max(3, j.pct) + '%';
  if (j.state === 'running') {
    $('#trainMsg').textContent = `Training… ${j.pct}% (${j.elapsed}s). You can keep using the app.`; $('#btnRetrain').disabled = true;
    trainTimer = setTimeout(pollTrain, 2500);
  } else {
    $('#btnRetrain').disabled = false;
    if (j.state === 'done') { $('#trainMsg').textContent = j.msg; await refreshCfg(); toast('New model is live'); }
    else if (j.state === 'error') $('#trainMsg').textContent = 'Training failed: ' + j.msg;
  }
}
async function onRetrain() {
  if (!confirm('Retrain on all recorded clips now? It takes a few minutes.')) return;
  const j = await api('retrain', {});
  if (!j.started) { $('#trainMsg').textContent = j.error || 'Could not start'; return; }
  $('#trainMsg').textContent = 'Starting…'; clearTimeout(trainTimer); pollTrain();
}

/* ---------- settings ---------- */
function bindSettings() {
  $('#setCam').value = S.facing; $('#setRes').value = String(S.res); $('#setCd').value = String(S.cd);
  $('#setVol').value = S.vol; $('#volTxt').textContent = Math.round(S.vol * 100) + '%';
  $('#setBeep').checked = S.beep; $('#setGpu').checked = S.gpu;
  $('#btnSettings').onclick = () => $('#dlg').showModal();
  $('#setCam').onchange = async (e) => { S.facing = e.target.value; store.set('cam', S.facing); await restartCam(); };
  $('#setRes').onchange = async (e) => { S.res = +e.target.value; store.set('res', S.res); await restartCam(); };
  $('#setCd').onchange = (e) => { S.cd = +e.target.value; store.set('cd', S.cd); };
  $('#setVoice').onchange = (e) => { S.voiceURI = e.target.value; store.set('voice', S.voiceURI); pickVoice(); };
  $('#setVol').oninput = (e) => { S.vol = +e.target.value; store.set('vol', S.vol); $('#volTxt').textContent = Math.round(S.vol * 100) + '%'; };
  $('#setBeep').onchange = (e) => { S.beep = e.target.checked; store.set('beep', S.beep); };
  $('#setGpu').onchange = (e) => { S.gpu = e.target.checked; store.set('gpu', S.gpu); };
  $('#btnTestVoice').onclick = () => { primeSpeech(); speak('Hello, I am SignTalk'); };
  $('#btnReload').onclick = () => location.reload();
  $('#btnMute').onclick = () => {
    S.speakOn = !S.speakOn; store.set('speak', S.speakOn); $('#btnMute').textContent = S.speakOn ? '🔊 Voice on' : '🔇 Voice off';
    if (!S.speakOn) speechSynthesis?.cancel();
  };
  $('#btnMute').textContent = S.speakOn ? '🔊 Voice on' : '🔇 Voice off';
  document.querySelectorAll('.tab').forEach((t) => (t.onclick = () => {
    document.querySelectorAll('.tab').forEach((x) => x.classList.toggle('on', x === t));
    document.querySelectorAll('.view').forEach((v) => v.classList.toggle('on', v.id === 'v-' + t.dataset.tab));
    S.tab = t.dataset.tab;
  }));
  $('#teachSel').onchange = teachCount;
  $('#btnSign').onclick = () => (S.autoOn ? togglePause() : onSign());
  $('#btnTeach').onclick = onTeach; $('#btnRetrain').onclick = onRetrain;
  $('#modeSeg').onclick = (e) => {
    const m = e.target.closest('button'); if (!m) return;
    S.autoOn = m.dataset.m === 'auto'; S.paused = false; store.set('auto', S.autoOn); applyMode();
  };
  $('#setRaise').value = String(S.raiseThr);
  $('#setRaise').onchange = (e) => { S.raiseThr = +e.target.value; store.set('raise', S.raiseThr); };
  $('#setThr').oninput = (e) => { S.thr = +e.target.value; store.set('thr', S.thr); $('#thrTxt').textContent = Math.round(S.thr * 100) + '%'; };
  $('#btnUndo').onclick = async () => {
    const j = await api('teach_undo', {});
    $('#teachMsg').textContent = j.undone ? `Removed the last clip of "${j.phrase}".` : 'Nothing to undo.'; await refreshCfg();
  };
}
async function restartCam() {
  S.ready = false; loadMsg('Switching camera…');
  try { await startCamera(); S.ready = true; $('#loading').classList.add('off'); } catch (e) { loadMsg(e.message, true); }
  syncButtons();
}
function syncButtons() { $('#btnSign').disabled = !S.ready; $('#btnTeach').disabled = !S.ready; }

/* ---------- wake lock ---------- */
async function keepAwake() {
  try { if ('wakeLock' in navigator && document.visibilityState === 'visible') S.wake = await navigator.wakeLock.request('screen'); } catch { /* optional */ }
}
document.addEventListener('visibilitychange', keepAwake);

/* ---------- boot ---------- */
(async function boot() {
  bindSettings();
  fit();
  ping();
  try {
    await refreshCfg();
    if (S.thr === null) S.thr = S.cfg.threshold;
    $('#setThr').value = S.thr; $('#thrTxt').textContent = Math.round(S.thr * 100) + '%';
    $('#aboutTxt').textContent = `${S.cfg.labels.length} phrases on this Pi.`;
    applyMode();
    await startCamera();
    await loadModels();
    S.ready = true; startLoop(); syncButtons(); applyMode(); keepAwake();
    $('#loading').classList.add('off');
    $('#aboutTxt').textContent += ` Tracking on ${S.delegate}.`;
    const q = new URLSearchParams(location.search);
    if (q.get('selftest') === 'files') await selftestFiles(q.get('clips').split(','), +q.get('fps') || 11.5);
    else if (q.has('selftest')) await selftest();
  } catch (e) {
    rlog('boot failed: ' + (e.stack || e)); loadMsg(String(e.message || e), true);
  }
})();

/* ---------- headless self-test (?selftest=1): records one window without taps and reports to the Pi log ---------- */
async function selftest() {
  await sleep(1500);
  const t0 = performance.now();
  const frames = await capture(0);
  const j = await api('predict', { aspect: S.ar, frames });
  rlog(`SELFTEST frames=${frames.length} fps~${Math.round(frames.length / ((performance.now() - t0) / 1000 - 0.1 || 1))} ` +
    `delegate=${S.delegate} result=${JSON.stringify(j)}`);
  S.mode = 'idle'; document.title = 'SELFTEST DONE';
}

/* ---------- parity test (?selftest=files&clips=/_test/a.mp4,...&fps=11.5): the same recorded clips through this page's tracker ---------- */
async function selftestFiles(urls, fps) {
  S.ready = false;                 // stop the live camera loop: MediaPipe needs one increasing timestamp stream
  let ts = S.lastTs + 1000;
  for (const url of urls) {
    const v = document.createElement('video');
    v.muted = true; v.playsInline = true; v.src = url;
    await new Promise((r, j) => { v.onloadedmetadata = r; v.onerror = () => j(new Error('cannot load ' + url)); });
    const frames = [];
    const n = Math.round(v.duration * fps);
    for (let i = 0; i < n; i++) {
      const shown = new Promise((r) => v.requestVideoFrameCallback(() => r()));   // wait until the new frame is really presented
      v.currentTime = Math.min(v.duration - 0.001, i / fps);
      await Promise.race([shown, sleep(1500)]);
      await sleep(20);
      ts += Math.round(1000 / fps);
      const hh = S.hands.detectForVideo(v, ts);
      if (new URLSearchParams(location.search).has('dump')) {
        rlog(`FRAME ${i} ` + (hh.landmarks || []).map((l, k) => `${hh.handedness[k][0].categoryName}:${hh.handedness[k][0].score.toFixed(2)}@${l[0].x.toFixed(2)},${l[0].y.toFixed(2)}`).join(' | '));
      }
      frames.push(pack(hh, S.pose.detectForVideo(v, ts)));
    }
    const j = await api('predict', { aspect: v.videoWidth / v.videoHeight, frames });
    const hands = frames.filter((f) => f[1] || f[2]).length;
    const mean = (a) => (a.length ? [r4(a.reduce((x, y) => x + y[0], 0) / a.length), r4(a.reduce((x, y) => x + y[1], 0) / a.length)] : null);
    rlog(`STATS ${url} lh=${frames.filter((f) => f[1]).length} rh=${frames.filter((f) => f[2]).length} pose=${frames.filter((f) => f[0]).length} `
      + `lhWrist=${JSON.stringify(mean(frames.filter((f) => f[1]).map((f) => f[1].slice(0, 2))))} rhWrist=${JSON.stringify(mean(frames.filter((f) => f[2]).map((f) => f[2].slice(0, 2))))} `
      + `nose=${JSON.stringify(mean(frames.filter((f) => f[0]).map((f) => f[0].slice(0, 2))))} lShoulder=${JSON.stringify(mean(frames.filter((f) => f[0]).map((f) => f[0].slice(2, 4))))}`);
    rlog(`PARITY ${url} frames=${frames.length} hands=${hands} -> ${j.best ? j.best.id + ' ' + j.best.p : j.reason} ${j.accepted ? 'accepted' : 'rejected'}`
      + (j.top ? ' | ' + j.top.slice(1).map((t) => t.id + ' ' + t.p).join(' ') : ''));
  }
  document.title = 'SELFTEST DONE'; rlog('PARITY DONE');
}
