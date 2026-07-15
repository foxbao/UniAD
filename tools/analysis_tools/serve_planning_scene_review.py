#!/usr/bin/env python3
"""Serve a local click-through UI for planning scene manifest review."""

import argparse
import csv
import json
import mimetypes
import os
import os.path as osp
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse


LABELS = ('NaturalRun', 'OperationalStop', 'DetectionProbe', 'Uncertain')
CONTROL_MODES = ('Auto', 'Manual', 'Mixed', 'Unknown')


INDEX_HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Planning Scene Review</title>
<style>
:root {
  color-scheme: light;
  --bg: #f4f5f6;
  --surface: #ffffff;
  --line: #d7dadd;
  --text: #202428;
  --muted: #667078;
  --accent: #146c5a;
  --accent-soft: #dcefe9;
  --warn: #9b5c00;
  --danger: #9a3030;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
button, input, select, textarea { font: inherit; }
button { cursor: pointer; }
.topbar {
  height: 58px;
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 0 18px;
  border-bottom: 1px solid var(--line);
  background: var(--surface);
}
h1 { margin: 0 14px 0 0; font-size: 17px; font-weight: 650; }
.filters { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
select, input, textarea {
  border: 1px solid #bcc2c7;
  border-radius: 4px;
  background: #fff;
  color: var(--text);
}
select, input { height: 34px; padding: 0 9px; }
.progress { margin-left: auto; color: var(--muted); white-space: nowrap; }
.workspace {
  min-height: calc(100vh - 58px);
  display: grid;
  grid-template-columns: minmax(520px, 1fr) 390px;
}
.visual {
  min-width: 0;
  display: flex;
  flex-direction: column;
  align-items: stretch;
  padding: 18px;
  background: #e8eaec;
}
.visual-tabs { display: flex; gap: 6px; margin-bottom: 8px; }
.visual-tab {
  height: 34px;
  padding: 0 13px;
  border: 1px solid #aeb5ba;
  border-radius: 4px;
  background: #fff;
}
.visual-tab.active { border-color: var(--accent); background: var(--accent); color: #fff; }
.visual-content { flex: 1; min-height: 0; display: grid; place-items: center; }
.visual-content img {
  display: block;
  width: 100%;
  height: calc(100vh - 136px);
  object-fit: contain;
  background: #fff;
  border: 1px solid var(--line);
}
.empty { color: var(--muted); font-size: 16px; }
.panel {
  min-width: 0;
  padding: 18px;
  border-left: 1px solid var(--line);
  background: var(--surface);
  overflow-y: auto;
}
.scene-head { margin-bottom: 14px; }
.scene-token { font: 600 13px/1.4 ui-monospace, SFMono-Regular, Consolas, monospace; overflow-wrap: anywhere; }
.scene-index { color: var(--muted); margin-top: 4px; }
.auto-summary {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 6px 14px;
  padding: 12px 0;
  border-top: 1px solid var(--line);
  border-bottom: 1px solid var(--line);
  font-size: 13px;
}
.auto-summary span { color: var(--muted); }
.reasons { grid-column: 1 / -1; overflow-wrap: anywhere; }
.field { padding: 14px 0; border-bottom: 1px solid var(--line); }
.field-label { display: block; margin-bottom: 8px; font-weight: 650; }
.segments { display: grid; gap: 6px; }
.segments.labels { grid-template-columns: 1fr 1fr; }
.segments.modes { grid-template-columns: repeat(4, 1fr); }
.segments.usable { grid-template-columns: 1fr 1fr; }
.choice {
  min-height: 38px;
  padding: 7px 8px;
  border: 1px solid #b9c0c5;
  border-radius: 4px;
  background: #fff;
  color: #32383d;
}
.choice.active { border-color: var(--accent); background: var(--accent-soft); color: #0d5547; font-weight: 650; }
.choice.danger.active { border-color: var(--danger); background: #f6e3e3; color: #772525; }
textarea { width: 100%; min-height: 72px; resize: vertical; padding: 8px; }
.reviewer-row { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.reviewer-row input { width: 100%; }
.actions { display: grid; grid-template-columns: 38px 1fr 1.4fr 38px; gap: 8px; padding-top: 16px; }
.action {
  height: 40px;
  border: 1px solid #aeb5ba;
  border-radius: 4px;
  background: #fff;
  color: var(--text);
}
.action.primary { border-color: var(--accent); background: var(--accent); color: #fff; font-weight: 650; }
.action:disabled { cursor: default; opacity: .45; }
.message { min-height: 21px; margin-top: 9px; color: var(--muted); }
.message.error { color: var(--danger); }
@media (max-width: 900px) {
  .topbar { height: auto; min-height: 58px; padding: 10px 12px; flex-wrap: wrap; }
  .progress { width: 100%; margin-left: 0; }
  .workspace { grid-template-columns: 1fr; }
  .visual { padding: 8px; }
  .visual-content img { height: auto; max-height: 65vh; }
  .panel { border-left: 0; border-top: 1px solid var(--line); }
}
</style>
</head>
<body>
<header class="topbar">
  <h1>Planning Scene Review</h1>
  <div class="filters">
    <select id="split" aria-label="Split">
      <option value="val">Val</option>
      <option value="train">Train</option>
      <option value="all">All splits</option>
    </select>
    <select id="status" aria-label="Review status">
      <option value="pending">Pending</option>
      <option value="reviewed">Reviewed</option>
      <option value="all">All states</option>
    </select>
    <select id="autoLabel" aria-label="Automatic label">
      <option value="all">All suggestions</option>
      <option value="DetectionProbe">DetectionProbe</option>
      <option value="Uncertain">Uncertain</option>
      <option value="NaturalRun">NaturalRun</option>
    </select>
  </div>
  <div class="progress" id="progress">Loading...</div>
</header>
<main class="workspace">
  <section class="visual">
    <div class="visual-tabs">
      <button class="visual-tab active" id="cameraTab">Camera</button>
      <button class="visual-tab" id="trajectoryTab">Trajectory</button>
    </div>
    <div class="visual-content" id="visual"><div class="empty">Loading...</div></div>
  </section>
  <aside class="panel">
    <div class="scene-head">
      <div class="scene-token" id="sceneToken">-</div>
      <div class="scene-index" id="sceneIndex">-</div>
    </div>
    <div class="auto-summary">
      <div><span>Suggestion</span><br><strong id="autoValue">-</strong></div>
      <div><span>Confidence</span><br><strong id="confidence">-</strong></div>
      <div><span>Parsed mode</span><br><strong id="parsedMode">-</strong></div>
      <div><span>Speed proxy</span><br><strong id="proxy">-</strong></div>
      <div class="reasons"><span>Mode sequence</span><br><strong id="modeSequence">-</strong></div>
      <div><span>Target</span><br><strong id="target">-</strong></div>
      <div class="reasons"><span>Signals</span><br><strong id="reasons">-</strong></div>
    </div>
    <div class="field">
      <span class="field-label">Scene type</span>
      <div class="segments labels" id="labelChoices"></div>
    </div>
    <div class="field">
      <span class="field-label">Control mode</span>
      <div class="segments modes" id="modeChoices"></div>
    </div>
    <div class="field">
      <span class="field-label">Planning usable</span>
      <div class="segments usable" id="usableChoices"></div>
    </div>
    <div class="field">
      <div class="reviewer-row">
        <label><span class="field-label">Reviewer</span><input id="reviewer" autocomplete="name"></label>
        <label><span class="field-label">Status</span><input id="savedStatus" disabled></label>
      </div>
    </div>
    <div class="field">
      <label><span class="field-label">Notes</span><textarea id="notes"></textarea></label>
    </div>
    <div class="actions">
      <button class="action" id="previous" title="Previous scene">&larr;</button>
      <button class="action" id="suggest">Prefill</button>
      <button class="action primary" id="save">Save and next</button>
      <button class="action" id="next" title="Next scene">&rarr;</button>
    </div>
    <div class="message" id="message"></div>
  </aside>
</main>
<script>
const labels = ['NaturalRun', 'OperationalStop', 'DetectionProbe', 'Uncertain'];
const modes = ['Auto', 'Manual', 'Mixed', 'Unknown'];
const state = { scenes: [], index: 0, draft: {}, visualMode: 'camera' };

function el(id) { return document.getElementById(id); }
function selectedScene() { return state.scenes[state.index] || null; }

function makeChoices(container, values, field, dangerValue) {
  container.replaceChildren();
  for (const value of values) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'choice' + (value === dangerValue ? ' danger' : '');
    button.textContent = value;
    button.dataset.value = value;
    button.addEventListener('click', () => {
      state.draft[field] = value;
      renderChoices();
    });
    container.appendChild(button);
  }
}

function renderChoices() {
  for (const button of el('labelChoices').children) {
    button.classList.toggle('active', button.dataset.value === state.draft.human_label);
  }
  for (const button of el('modeChoices').children) {
    button.classList.toggle('active', button.dataset.value === state.draft.human_control_mode);
  }
  for (const button of el('usableChoices').children) {
    button.classList.toggle('active', button.dataset.value === state.draft.planning_usable);
  }
}

function proxyMode(proxy) {
  if (proxy === 'LikelyAuto') return 'Auto';
  if (proxy === 'LikelyManual') return 'Manual';
  return 'Unknown';
}

function loadDraft(scene) {
  const parsedMode = modes.includes(scene.parsed_control_mode) && scene.parsed_control_mode !== 'Unknown'
    ? scene.parsed_control_mode : '';
  state.draft = {
    human_label: scene.human_label || '',
    human_control_mode: scene.human_control_mode || parsedMode,
    planning_usable: scene.planning_usable || '',
  };
  el('reviewer').value = scene.reviewer || localStorage.getItem('planningReviewer') || '';
  el('notes').value = scene.notes || '';
}

function renderMedia(scene) {
  const visual = el('visual');
  visual.replaceChildren();
  if (!scene) {
    const empty = document.createElement('div');
    empty.className = 'empty';
    empty.textContent = 'No scenes in this queue.';
    visual.appendChild(empty);
    return;
  }
  let mediaPath = state.visualMode === 'camera'
    ? scene.camera_contact_sheet : scene.plot_path;
  if (!mediaPath && state.visualMode === 'camera') mediaPath = scene.plot_path;
  if (mediaPath) {
    const image = document.createElement('img');
    image.src = '/asset/' + mediaPath.split('/').map(encodeURIComponent).join('/');
    image.alt = scene.scene_token;
    visual.appendChild(image);
  } else {
    const empty = document.createElement('div');
    empty.className = 'empty';
    empty.textContent = 'No plot generated for this scene.';
    visual.appendChild(empty);
  }
  el('cameraTab').classList.toggle('active', state.visualMode === 'camera');
  el('trajectoryTab').classList.toggle('active', state.visualMode === 'trajectory');
}

function render() {
  const scene = selectedScene();
  renderMedia(scene);
  if (!scene) {
    el('sceneToken').textContent = '-';
    el('sceneIndex').textContent = '-';
    el('save').disabled = true;
    return;
  }
  el('sceneToken').textContent = scene.split + ' / ' + scene.scene_token;
  el('sceneIndex').textContent = `${state.index + 1} of ${state.scenes.length} - ${scene.frame_count} frames`;
  el('autoValue').textContent = scene.auto_label || '-';
  el('confidence').textContent = Number(scene.auto_confidence || 0).toFixed(2);
  el('parsedMode').textContent = scene.parsed_control_mode || '-';
  el('proxy').textContent = scene.control_mode_proxy || '-';
  el('modeSequence').textContent = scene.control_mode_sequence || '-';
  el('target').textContent = scene.dominant_target_class || 'none';
  el('reasons').textContent = (scene.reason_codes || 'none').replaceAll(';', '; ');
  el('savedStatus').value = scene.review_status || 'pending';
  el('previous').disabled = state.index === 0;
  el('next').disabled = state.index >= state.scenes.length - 1;
  el('save').disabled = false;
  loadDraft(scene);
  renderChoices();
  el('message').textContent = '';
  el('message').className = 'message';
}

async function refresh() {
  const params = new URLSearchParams({
    split: el('split').value,
    status: el('status').value,
    auto_label: el('autoLabel').value,
  });
  const response = await fetch('/api/scenes?' + params);
  const payload = await response.json();
  state.scenes = payload.scenes;
  state.index = 0;
  el('progress').textContent = `${payload.summary.reviewed}/${payload.summary.total} reviewed - ${payload.scenes.length} shown`;
  render();
}

function useSuggestion() {
  const scene = selectedScene();
  if (!scene) return;
  state.draft.human_label = scene.auto_label || 'Uncertain';
  state.draft.human_control_mode = modes.includes(scene.parsed_control_mode)
    ? scene.parsed_control_mode : proxyMode(scene.control_mode_proxy);
  state.draft.planning_usable = scene.auto_label === 'NaturalRun' ? '1' : '0';
  renderChoices();
}

async function save() {
  const scene = selectedScene();
  if (!scene) return;
  if (!state.draft.human_label || !state.draft.human_control_mode || state.draft.planning_usable === '') {
    el('message').textContent = 'Scene type, control mode, and planning usability are required.';
    el('message').className = 'message error';
    return;
  }
  const reviewer = el('reviewer').value.trim();
  if (reviewer) localStorage.setItem('planningReviewer', reviewer);
  el('save').disabled = true;
  const response = await fetch('/api/review', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      split: scene.split,
      scene_token: scene.scene_token,
      ...state.draft,
      reviewer,
      notes: el('notes').value,
    }),
  });
  const payload = await response.json();
  if (!response.ok) {
    el('message').textContent = payload.error || 'Save failed.';
    el('message').className = 'message error';
    el('save').disabled = false;
    return;
  }
  if (el('status').value === 'pending') {
    state.scenes.splice(state.index, 1);
    if (state.index >= state.scenes.length) state.index = Math.max(0, state.scenes.length - 1);
  } else {
    Object.assign(scene, payload.scene);
    if (state.index < state.scenes.length - 1) state.index += 1;
  }
  el('progress').textContent = `${payload.summary.reviewed}/${payload.summary.total} reviewed - ${state.scenes.length} shown`;
  render();
}

makeChoices(el('labelChoices'), labels, 'human_label', 'DetectionProbe');
makeChoices(el('modeChoices'), modes, 'human_control_mode');
makeChoices(el('usableChoices'), ['1', '0'], 'planning_usable', '0');
el('usableChoices').children[0].textContent = 'Yes';
el('usableChoices').children[1].textContent = 'No';
for (const id of ['split', 'status', 'autoLabel']) el(id).addEventListener('change', refresh);
el('previous').addEventListener('click', () => { if (state.index > 0) { state.index -= 1; render(); } });
el('next').addEventListener('click', () => { if (state.index + 1 < state.scenes.length) { state.index += 1; render(); } });
el('suggest').addEventListener('click', useSuggestion);
el('save').addEventListener('click', save);
el('cameraTab').addEventListener('click', () => { state.visualMode = 'camera'; renderMedia(selectedScene()); });
el('trajectoryTab').addEventListener('click', () => { state.visualMode = 'trajectory'; renderMedia(selectedScene()); });
refresh().catch(error => {
  el('message').textContent = error.message;
  el('message').className = 'message error';
});
</script>
</body>
</html>
'''


class ManifestStore:

    def __init__(self, audit_dir):
        self.audit_dir = osp.realpath(audit_dir)
        self.manifest_path = osp.join(self.audit_dir, 'scene_manifest.csv')
        self.queue_path = osp.join(self.audit_dir, 'review_queue.csv')
        self.lock = threading.Lock()

    @staticmethod
    def _read_csv(path):
        with open(path, newline='', encoding='utf-8') as handle:
            reader = csv.DictReader(handle)
            return list(reader), list(reader.fieldnames or [])

    def list_scenes(self, split='val', status='pending', auto_label='all'):
        with self.lock:
            rows, _ = self._read_csv(self.manifest_path)
            queue_rows, _ = self._read_csv(self.queue_path)
        queue_order = {
            (row.get('split', ''), row.get('scene_token', '')): index
            for index, row in enumerate(queue_rows)
        }
        rows.sort(key=lambda row: queue_order.get(
            (row.get('split', ''), row.get('scene_token', '')), 10 ** 9))
        split_rows = rows if split == 'all' else [
            row for row in rows if row.get('split') == split]
        reviewed = sum(
            row.get('review_status') == 'reviewed' for row in split_rows)
        filtered = split_rows
        if status == 'pending':
            filtered = [
                row for row in filtered
                if row.get('review_status') != 'reviewed']
        elif status == 'reviewed':
            filtered = [
                row for row in filtered
                if row.get('review_status') == 'reviewed']
        if auto_label != 'all':
            filtered = [
                row for row in filtered
                if row.get('auto_label') == auto_label]
        return filtered, dict(total=len(split_rows), reviewed=reviewed)

    def review(self, payload):
        required = ('split', 'scene_token', 'human_label',
                    'human_control_mode', 'planning_usable')
        missing = [key for key in required if str(payload.get(key, '')) == '']
        if missing:
            raise ValueError(f'Missing fields: {", ".join(missing)}')
        if payload['human_label'] not in LABELS:
            raise ValueError(f'Invalid human_label: {payload["human_label"]}')
        if payload['human_control_mode'] not in CONTROL_MODES:
            raise ValueError(
                f'Invalid human_control_mode: {payload["human_control_mode"]}')
        usable = str(payload['planning_usable'])
        if usable not in ('0', '1'):
            raise ValueError('planning_usable must be 0 or 1')
        key = (str(payload['split']), str(payload['scene_token']))
        with self.lock:
            rows, fieldnames = self._read_csv(self.manifest_path)
            matched = None
            for row in rows:
                if (row.get('split'), row.get('scene_token')) == key:
                    row.update(
                        human_label=payload['human_label'],
                        human_control_mode=payload['human_control_mode'],
                        planning_usable=usable,
                        review_status='reviewed',
                        reviewer=str(payload.get('reviewer', ''))[:200],
                        notes=str(payload.get('notes', ''))[:4000],
                    )
                    matched = dict(row)
                    break
            if matched is None:
                raise KeyError(f'Scene not found: {key}')
            self._write_csv_atomic(rows, fieldnames)
            split_rows = [row for row in rows if row.get('split') == key[0]]
            summary = dict(
                total=len(split_rows),
                reviewed=sum(
                    row.get('review_status') == 'reviewed'
                    for row in split_rows),
            )
        return matched, summary

    def _write_csv_atomic(self, rows, fieldnames):
        descriptor, temporary = tempfile.mkstemp(
            prefix='.scene_manifest.', suffix='.tmp', dir=self.audit_dir)
        try:
            with os.fdopen(
                    descriptor, 'w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.manifest_path)
        except Exception:
            if osp.exists(temporary):
                os.unlink(temporary)
            raise


def make_handler(store):

    class ReviewHandler(BaseHTTPRequestHandler):

        def send_bytes(self, payload, content_type, status=200):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(payload)

        def send_json(self, payload, status=200):
            body = json.dumps(payload).encode('utf-8')
            self.send_bytes(body, 'application/json; charset=utf-8', status)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == '/':
                self.send_bytes(
                    INDEX_HTML.encode('utf-8'), 'text/html; charset=utf-8')
                return
            if parsed.path == '/api/scenes':
                query = parse_qs(parsed.query)
                scenes, summary = store.list_scenes(
                    split=query.get('split', ['val'])[0],
                    status=query.get('status', ['pending'])[0],
                    auto_label=query.get('auto_label', ['all'])[0],
                )
                self.send_json(dict(scenes=scenes, summary=summary))
                return
            if parsed.path.startswith('/asset/'):
                relative = unquote(parsed.path[len('/asset/'):])
                target = osp.realpath(osp.join(store.audit_dir, relative))
                if (osp.commonpath([store.audit_dir, target])
                        != store.audit_dir or not osp.isfile(target)):
                    self.send_json(dict(error='Asset not found'), 404)
                    return
                with open(target, 'rb') as handle:
                    payload = handle.read()
                content_type = mimetypes.guess_type(target)[0] \
                    or 'application/octet-stream'
                self.send_bytes(payload, content_type)
                return
            self.send_json(dict(error='Not found'), 404)

        def do_POST(self):
            if urlparse(self.path).path != '/api/review':
                self.send_json(dict(error='Not found'), 404)
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if length <= 0 or length > 1024 * 1024:
                    raise ValueError('Invalid request size')
                payload = json.loads(self.rfile.read(length))
                scene, summary = store.review(payload)
                self.send_json(dict(scene=scene, summary=summary))
            except KeyError as error:
                self.send_json(dict(error=str(error)), 404)
            except (ValueError, json.JSONDecodeError) as error:
                self.send_json(dict(error=str(error)), 400)
            except Exception as error:
                self.send_json(dict(error=str(error)), 500)

        def log_message(self, format_, *args):
            print(f'{self.client_address[0]} - {format_ % args}')

    return ReviewHandler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit-dir', required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    return parser.parse_args()


def main():
    args = parse_args()
    store = ManifestStore(args.audit_dir)
    if not osp.isfile(store.manifest_path):
        raise FileNotFoundError(store.manifest_path)
    if not osp.isfile(store.queue_path):
        raise FileNotFoundError(store.queue_path)
    server = ThreadingHTTPServer(
        (args.host, args.port), make_handler(store))
    print(f'Planning scene review: http://{args.host}:{args.port}')
    print(f'Manifest: {store.manifest_path}')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
