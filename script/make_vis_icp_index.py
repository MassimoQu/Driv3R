import argparse
import json
import shutil
import urllib.error
import urllib.request
from pathlib import Path

_THREE_VERSION = "0.160.0"
_VIEWER_FILES = {
    "three.module.js": f"https://unpkg.com/three@{_THREE_VERSION}/build/three.module.js",
    "OrbitControls.js": f"https://unpkg.com/three@{_THREE_VERSION}/examples/jsm/controls/OrbitControls.js",
    "PLYLoader.js": f"https://unpkg.com/three@{_THREE_VERSION}/examples/jsm/loaders/PLYLoader.js",
}


def _safe_read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _fmt(v, nd=3):
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return "-"


def _html_escape(s: str) -> str:
    s = str(s)
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _ensure_viewer_deps(root: Path):
    """
    Ensure `<root>/_viewer/{three.module.js,OrbitControls.js,PLYLoader.js}` exists.
    Will try to copy from Driv3R/output/_viewer_cache; if missing, downloads from unpkg.
    """
    vdir = root / "_viewer"
    vdir.mkdir(parents=True, exist_ok=True)

    cache = Path(__file__).resolve().parents[1] / "output" / "_viewer_cache"
    cache.mkdir(parents=True, exist_ok=True)

    for name, url in _VIEWER_FILES.items():
        cache_file = cache / name
        if not cache_file.exists() or cache_file.stat().st_size < 1024:
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    cache_file.write_bytes(r.read())
            except (urllib.error.URLError, TimeoutError):
                # If download fails, the viewer will show a helpful message.
                pass

        dst = vdir / name
        if cache_file.exists() and (not dst.exists() or dst.stat().st_size != cache_file.stat().st_size):
            shutil.copyfile(cache_file, dst)


def _write_viewer(root: Path, title: str):
    _ensure_viewer_deps(root)
    viewer_path = root / "viewer.html"
    safe_title = _html_escape(title)

    template = r"""<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>__TITLE__ - 交互式点云</title>
    <style>
      html, body { height: 100%; margin: 0; background: #0b1020; }
      #app { position: fixed; inset: 0; display: grid; grid-template-rows: auto 1fr; }
      #topbar {
        display: flex; flex-wrap: wrap; gap: 10px; align-items: center;
        padding: 10px 12px; color: #e5e7eb; background: rgba(10, 15, 30, 0.85);
        border-bottom: 1px solid rgba(255,255,255,0.08);
        font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Noto Sans", "Liberation Sans";
        font-size: 13px;
        backdrop-filter: blur(8px);
      }
      #topbar a { color: #93c5fd; text-decoration: none; }
      #topbar a:hover { text-decoration: underline; }
      #topbar .spacer { flex: 1; }
      #topbar .chip {
        display: inline-flex; gap: 6px; align-items: center;
        padding: 4px 8px; border: 1px solid rgba(255,255,255,0.10); border-radius: 999px;
        background: rgba(255,255,255,0.04);
      }
      #topbar input[type="range"] { width: 140px; }
      #topbar input[type="checkbox"] { transform: translateY(1px); }
      #topbar button {
        padding: 6px 10px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.12);
        background: rgba(255,255,255,0.06); color: #e5e7eb; cursor: pointer;
      }
      #topbar button:hover { background: rgba(255,255,255,0.10); }
      #canvas-wrap { position: relative; }
      #hud {
        position: absolute; left: 12px; bottom: 12px;
        padding: 10px 12px; border-radius: 12px;
        color: #e5e7eb; background: rgba(10, 15, 30, 0.60);
        border: 1px solid rgba(255,255,255,0.08);
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New";
        font-size: 12px; line-height: 1.35;
        max-width: min(720px, calc(100vw - 24px));
        white-space: pre-wrap;
      }
      #status {
        position: absolute; right: 12px; bottom: 12px;
        padding: 10px 12px; border-radius: 12px;
        color: #e5e7eb; background: rgba(10, 15, 30, 0.60);
        border: 1px solid rgba(255,255,255,0.08);
        font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
        font-size: 12px;
      }
    </style>
  </head>
  <body>
    <div id="app">
      <div id="topbar">
        <span class="chip"><b>Driv3R</b> 交互式点云</span>
        <span class="chip">scene: <span id="scene-name">-</span></span>
        <span class="chip">切换:
          <select id="scene-select" style="max-width: 520px;"></select>
        </span>
        <label class="chip"><input id="toggle-gt" type="checkbox" checked /> GT</label>
        <label class="chip"><input id="toggle-pred" type="checkbox" checked /> Pred</label>
        <label class="chip">point size <input id="pt-size" type="range" min="0.3" max="6.0" step="0.1" value="1.6" /></label>
        <label class="chip"><input id="toggle-axes" type="checkbox" /> axes</label>
        <button id="btn-fit">Fit</button>
        <button id="btn-reset">Reset</button>
        <button id="btn-shot">Screenshot</button>
        <span class="spacer"></span>
        <a href="index.html">Back to gallery</a>
      </div>
      <div id="canvas-wrap">
        <div id="hud" style="display:none"></div>
        <div id="status">Loading...</div>
      </div>
    </div>

    <script>
      // If opened via file:// (or module load fails), keep a helpful message instead of "Loading..." forever.
      window.addEventListener('error', () => {
        const el = document.getElementById('status');
        if (!el) return;
        el.style.display = 'block';
        el.textContent =
          'JS 加载失败：请在当前目录运行 `python3 -m http.server 8000` 后通过 http:// 访问本页，且确保存在 `_viewer/three.module.js`。';
      });
    </script>

    <script type="module">
      const statusEl = document.getElementById('status');
      const hudEl = document.getElementById('hud');
      const sceneNameEl = document.getElementById('scene-name');

      if (location.protocol === 'file:') {
        statusEl.style.display = 'block';
        statusEl.textContent =
          '当前是 file:// 打开，浏览器会阻止 ES Module 加载。请在目录下运行 `python3 -m http.server 8000` 后通过 http:// 打开本页。';
        throw new Error('Open via http.server instead of file://');
      }

      function q(name, fallback=null) {
        const v = new URLSearchParams(location.search).get(name);
        return (v === null || v === undefined || v === '') ? fallback : v;
      }

      function joinPath(base, rel) {
        if (!base) return rel;
        const b = base.endsWith('/') ? base : base + '/';
        return b + rel;
      }

      const sceneRel = (q('scene', '') || '').replace(/^\/+/, '').replace(/\/+$/, '');
      sceneNameEl.textContent = sceneRel || '(root)';
      const lite = String(q('lite', '1')).toLowerCase();
      const liteEnabled = !(lite === '0' || lite === 'false' || lite === 'no' || lite === 'off');

      const gtUrl = q('gt', joinPath(sceneRel, 'gt.ply'));
      const predUrl = q('pred', joinPath(sceneRel, 'pred_icp.ply'));
      const metricsUrl = q('metrics', joinPath(sceneRel, 'metrics.json'));
      const scenesUrl = q('scenes', 'scenes.json');

      function setStatus(msg) {
        statusEl.textContent = msg;
        statusEl.style.display = 'block';
      }

      function withPlySuffix(url, suffix) {
        try {
          const m = String(url).match(/^([^?#]+)(\\?[^#]*)?(#.*)?$/);
          if (!m) return null;
          const base = m[1] || '';
          const qs = m[2] || '';
          const hash = m[3] || '';
          if (!base.toLowerCase().endsWith('.ply')) return null;
          const lower = base.toLowerCase();
          if (lower.endsWith(`${suffix}.ply`)) return null;
          return base.slice(0, -4) + suffix + '.ply' + qs + hash;
        } catch (e) {
          return null;
        }
      }

      async function urlExists(url) {
        try {
          const r = await fetch(url, { method: 'HEAD', cache: 'no-store' });
          return !!(r && r.ok);
        } catch (e) {
          return false;
        }
      }

      async function preferLitePly(url) {
        if (!liteEnabled) return url;
        const cand = withPlySuffix(url, '_web');
        if (!cand) return url;
        const ok = await urlExists(cand);
        return ok ? cand : url;
      }

      async function tryFetchJson(url) {
        try {
          const r = await fetch(url);
          if (!r.ok) return null;
          return await r.json();
        } catch (e) {
          return null;
        }
      }

      function setSceneSelectOptions(scenes) {
        const select = document.getElementById('scene-select');
        select.innerHTML = '';
        const emptyOpt = document.createElement('option');
        emptyOpt.value = '';
        emptyOpt.textContent = '(choose scene...)';
        select.appendChild(emptyOpt);
        for (const s of scenes) {
          const opt = document.createElement('option');
          opt.value = s;
          opt.textContent = s;
          if (s === sceneRel) opt.selected = true;
          select.appendChild(opt);
        }
        select.addEventListener('change', (e) => {
          const v = e.target.value || '';
          if (!v) return;
          const next = new URL(location.href);
          next.searchParams.set('scene', v);
          next.searchParams.delete('gt');
          next.searchParams.delete('pred');
          next.searchParams.delete('metrics');
          location.href = next.toString();
        });
      }

      async function loadDeps() {
        setStatus('Loading viewer deps...');
        try {
          const THREE = await import('./_viewer/three.module.js');
          const { OrbitControls } = await import('./_viewer/OrbitControls.js');
          const { PLYLoader } = await import('./_viewer/PLYLoader.js');
          return { THREE, OrbitControls, PLYLoader };
        } catch (e) {
          console.error(e);
          setStatus('依赖加载失败：请确认用 http.server 打开，且目录下有 `_viewer/`（three.js / OrbitControls / PLYLoader）。');
          return null;
        }
      }

      async function run() {
        const deps = await loadDeps();
        if (!deps) return;
        const { THREE, OrbitControls, PLYLoader } = deps;

        const wrap = document.getElementById('canvas-wrap');
        const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
        renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
        renderer.setSize(wrap.clientWidth, wrap.clientHeight);
        renderer.setClearColor(0x0b1020, 1.0);
        wrap.appendChild(renderer.domElement);

        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(60, wrap.clientWidth / wrap.clientHeight, 0.01, 10000);
        camera.position.set(0, 0, 30);

        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.08;
        controls.screenSpacePanning = true;

        const axes = new THREE.AxesHelper(5);
        axes.visible = false;
        scene.add(axes);

        const grid = new THREE.GridHelper(40, 40, 0x334155, 0x1f2937);
        grid.material.opacity = 0.35;
        grid.material.transparent = true;
        scene.add(grid);

        let gtPoints = null;
        let predPoints = null;
        let lastBounds = null;

        function unionBounds(a, b) {
          if (!a) return b;
          if (!b) return a;
          const u = a.clone();
          u.union(b);
          return u;
        }

        function geometryBounds(geom) {
          geom.computeBoundingBox();
          const bb = geom.boundingBox;
          if (!bb) return null;
          return new THREE.Box3(bb.min.clone(), bb.max.clone());
        }

        function fitToBounds(bounds) {
          if (!bounds) return;
          const center = new THREE.Vector3();
          bounds.getCenter(center);
          const size = new THREE.Vector3();
          bounds.getSize(size);
          const radius = Math.max(size.x, size.y, size.z) * 0.5;
          const dist = radius / Math.tan((camera.fov * Math.PI / 180) * 0.5);
          const dir = new THREE.Vector3(1, 1, 1).normalize();
          camera.position.copy(center.clone().add(dir.multiplyScalar(dist * 1.2)));
          camera.near = Math.max(0.01, dist / 1000);
          camera.far = Math.max(1000, dist * 10);
          camera.updateProjectionMatrix();
          controls.target.copy(center);
          controls.update();
        }

        function makePoints(geom, defaultColorHex) {
          if (!geom.getAttribute('color')) {
            const n = geom.getAttribute('position').count;
            const colors = new Float32Array(n * 3);
            const c = new THREE.Color(defaultColorHex);
            for (let i = 0; i < n; i++) {
              colors[i*3+0] = c.r;
              colors[i*3+1] = c.g;
              colors[i*3+2] = c.b;
            }
            geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
          }
          const mat = new THREE.PointsMaterial({
            size: parseFloat(document.getElementById('pt-size').value),
            vertexColors: true,
            transparent: true,
            opacity: 1.0,
            depthWrite: false,
            sizeAttenuation: true,
          });
          return new THREE.Points(geom, mat);
        }

        function loadPLY(url) {
          return new Promise((resolve, reject) => {
            const loader = new PLYLoader();
            loader.load(
              url,
              (geom) => resolve(geom),
              (xhr) => {
                if (xhr && xhr.total) {
                  const p = Math.round((xhr.loaded / xhr.total) * 100);
                  setStatus(`Loading PLY... ${p}%`);
                }
              },
              (err) => reject(err)
            );
          });
        }

        function animate() {
          requestAnimationFrame(animate);
          controls.update();
          renderer.render(scene, camera);
        }
        animate();

        // UI bindings
        document.getElementById('toggle-gt').addEventListener('change', (e) => {
          if (gtPoints) gtPoints.visible = e.target.checked;
        });
        document.getElementById('toggle-pred').addEventListener('change', (e) => {
          if (predPoints) predPoints.visible = e.target.checked;
        });
        document.getElementById('toggle-axes').addEventListener('change', (e) => {
          axes.visible = e.target.checked;
        });
        document.getElementById('pt-size').addEventListener('input', (e) => {
          const v = parseFloat(e.target.value);
          if (gtPoints) gtPoints.material.size = v;
          if (predPoints) predPoints.material.size = v;
        });
        document.getElementById('btn-fit').addEventListener('click', () => fitToBounds(lastBounds));
        document.getElementById('btn-reset').addEventListener('click', () => {
          camera.position.set(0, 0, 30);
          controls.target.set(0, 0, 0);
          controls.update();
        });
        document.getElementById('btn-shot').addEventListener('click', () => {
          const a = document.createElement('a');
          a.download = (sceneRel ? sceneRel.replace(/\//g, '_') : 'scene') + '.png';
          a.href = renderer.domElement.toDataURL('image/png');
          a.click();
        });

        function onResize() {
          const w = wrap.clientWidth;
          const h = wrap.clientHeight;
          camera.aspect = w / h;
          camera.updateProjectionMatrix();
          renderer.setSize(w, h);
        }
        window.addEventListener('resize', onResize);

        // scene dropdown
        const scenesObj = await tryFetchJson(scenesUrl);
        const scenes = (scenesObj && Array.isArray(scenesObj.scenes)) ? scenesObj.scenes : null;
        if (scenes && scenes.length) setSceneSelectOptions(scenes);

        setStatus('Loading point clouds...');
        const resolvedGtUrl = await preferLitePly(gtUrl);
        const resolvedPredUrl = await preferLitePly(predUrl);
        const [gtGeom, predGeom] = await Promise.all([
          loadPLY(resolvedGtUrl).catch(() => null),
          loadPLY(resolvedPredUrl).catch(() => null),
        ]);
        if (!gtGeom && !predGeom) {
          const hint = liteEnabled ? ' (Also tried *_web.ply)' : '';
          setStatus('Failed to load gt/pred PLY' + hint + '. (Run with http.server; check file paths.)');
          return;
        }

        if (gtGeom) {
          gtPoints = makePoints(gtGeom, 0x9ca3af);
          scene.add(gtPoints);
          lastBounds = unionBounds(lastBounds, geometryBounds(gtGeom));
        }
        if (predGeom) {
          predPoints = makePoints(predGeom, 0x60a5fa);
          scene.add(predPoints);
          lastBounds = unionBounds(lastBounds, geometryBounds(predGeom));
        }

          const metrics = await tryFetchJson(metricsUrl);
        if (metrics) {
          const keys = [
            ['accuracy_raw', 'Acc raw'],
            ['accuracy', 'Acc'],
            ['completion_raw', 'Comp raw'],
            ['completion', 'Comp'],
            ['completion_ratio_raw', 'CR raw@0.2'],
            ['completion_ratio', 'CR@0.2'],
            ['icp_fitness', 'ICP fit'],
            ['icp_rmse', 'ICP rmse'],
            ['icp_rot_deg', 'ICP rot(deg)'],
            ['icp_centered_t_norm', 'ICP t(center)'],
            ['centroid_diff_norm', 'centroid Δ'],
            ['n_pred', 'N_pred'],
            ['n_gt', 'N_gt'],
          ];
          const lines = [];
          for (const [k, label] of keys) {
            if (metrics[k] !== undefined) {
              const v = (typeof metrics[k] === 'number') ? metrics[k].toFixed(4) : String(metrics[k]);
              lines.push(`${label}: ${v}`);
            }
          }
          if (lines.length) {
            hudEl.style.display = 'block';
            hudEl.textContent = lines.join('\n');
          }
        }

        fitToBounds(lastBounds);
        setStatus('Drag to rotate, right-drag to pan, wheel to zoom.');
        setTimeout(() => { statusEl.style.display = 'none'; }, 2500);
      }

      run().catch((e) => {
        console.error(e);
        setStatus('Error: ' + (e?.message || String(e)));
      });
    </script>
  </body>
</html>
"""

    viewer_path.write_text(template.replace("__TITLE__", safe_title), encoding="utf-8")
    return viewer_path


def main():
    ap = argparse.ArgumentParser("Generate an HTML gallery for ICP visualization outputs.")
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--title", type=str, default="Driv3R 可视化（ICP 对齐）")
    ap.add_argument("--video_name", type=str, default="overlay_orbit.mp4")
    ap.add_argument("--snap_prefix", type=str, default="overlay_view")
    ap.add_argument("--out", type=Path, default=None, help="Defaults to <root>/index.html")
    args = ap.parse_args()

    root = args.root
    out_path = args.out or (root / "index.html")

    # Always (re)write viewer.html and ensure _viewer deps.
    _write_viewer(root, args.title)

    runs = []
    for metrics_path in sorted(root.rglob("metrics.json")):
        run_dir = metrics_path.parent
        rel_dir = run_dir.relative_to(root)
        meta = _safe_read_json(metrics_path) or {}
        video = run_dir / args.video_name
        snaps = [run_dir / f"{args.snap_prefix}{i}.png" for i in range(5)]
        runs.append(
            dict(
                rel_dir=str(rel_dir),
                title=f"{rel_dir}",
                meta=meta,
                video=str(video.relative_to(root)) if video.exists() else None,
                snaps=[str(p.relative_to(root)) for p in snaps if p.exists()],
            )
        )

    scenes = sorted({r["rel_dir"] for r in runs})
    (root / "scenes.json").write_text(json.dumps({"scenes": scenes}, indent=2), encoding="utf-8")

    def card_html(run):
        meta = run["meta"] or {}
        metrics = [
            ("Acc raw", _fmt(meta.get("accuracy_raw"))),
            ("Acc", _fmt(meta.get("accuracy"))),
            ("Comp raw", _fmt(meta.get("completion_raw"))),
            ("Comp", _fmt(meta.get("completion"))),
            ("CR raw@0.2", _fmt(meta.get("completion_ratio_raw"))),
            ("CR@0.2", _fmt(meta.get("completion_ratio"))),
            ("ICP fit", _fmt(meta.get("icp_fitness"))),
            ("ICP rmse", _fmt(meta.get("icp_rmse"))),
            ("ICP rot(deg)", _fmt(meta.get("icp_rot_deg"))),
            ("ICP t(center)", _fmt(meta.get("icp_centered_t_norm"))),
            ("centroid Δ", _fmt(meta.get("centroid_diff_norm"))),
            ("N_pred", meta.get("n_pred", "-")),
            ("N_gt", meta.get("n_gt", "-")),
        ]
        metrics_html = " ".join([f"<span><b>{k}</b>: {v}</span>" for k, v in metrics])

        video_html = (
            f'<video controls src="{run["video"]}"></video>'
            if run["video"]
            else '<div class="hint">No video found.</div>'
        )
        viewer_link = f'<a class="viewer" target="_blank" href="viewer.html?scene={run["rel_dir"]}">新标签打开交互式点云</a>'
        viewer_load_btn = f'<button class="viewerbtn" type="button" data-scene="{run["rel_dir"]}">加载到本页 Viewer</button>'
        imgs_html = "".join([f'<img src="{p}" />' for p in run["snaps"]])
        return f"""
      <div class="card">
        <div class="title">{run["title"]}</div>
        <div class="metrics">{metrics_html}</div>
        <div class="viewer-wrap">{viewer_link}{viewer_load_btn}</div>
        {video_html}
        <div class="imgs">{imgs_html}</div>
      </div>
        """.rstrip()

    viewer_panel = ""
    if scenes:
        first_scene = scenes[0]
        viewer_panel = f"""
    <div class="viewerpanel">
      <div class="viewerbar">
        <div class="viewerbar-left">
          <span class="viewerbadge">交互式点云 Viewer</span>
          <label>scene
            <select id="viewerScene"></select>
          </label>
          <a id="viewerOpen" class="viewerlink" target="_blank" href="viewer.html?scene={first_scene}">新标签打开</a>
        </div>
        <div class="viewerbar-right">
          <span class="viewerhint">拖拽旋转 / 右键平移 / 滚轮缩放</span>
        </div>
      </div>
      <iframe id="viewerFrame" class="viewerframe" loading="lazy" src="viewer.html?scene={first_scene}"></iframe>
    </div>
        """.rstrip()

    index_template = """<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>__TITLE__</title>
    <style>
      body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Noto Sans", "Liberation Sans"; margin: 24px; }
      h1 { margin: 0 0 10px; }
      .hint { color: #6b7280; margin: 8px 0 18px; font-size: 12px; }
      .grid { display: grid; grid-template-columns: 1fr; gap: 16px; }
      .viewerpanel { border: 1px solid #e5e7eb; border-radius: 12px; overflow: hidden; margin: 0 0 18px; }
      .viewerbar { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 10px; padding: 10px 12px; border-bottom: 1px solid #e5e7eb; background: #f9fafb; align-items: center; }
      .viewerbar-left { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
      .viewerbar-right { display: flex; gap: 10px; align-items: center; }
      .viewerbadge { font-weight: 700; }
      .viewerhint { color: #6b7280; font-size: 12px; }
      .viewerframe { width: 100%; height: min(72vh, 820px); border: 0; background: #0b1020; }
      .card { padding: 14px; border: 1px solid #e5e7eb; border-radius: 12px; }
      .title { font-weight: 700; margin-bottom: 6px; }
      .metrics { display: flex; flex-wrap: wrap; gap: 10px; font-size: 12px; color: #111827; margin-bottom: 10px; }
      .viewer-wrap { margin: 4px 0 10px; }
      .viewer { display: inline-block; font-size: 12px; color: #2563eb; text-decoration: none; }
      .viewer:hover { text-decoration: underline; }
      .viewerlink { font-size: 12px; color: #2563eb; text-decoration: none; }
      .viewerlink:hover { text-decoration: underline; }
      .viewerbtn { margin-left: 10px; font-size: 12px; padding: 5px 8px; border-radius: 10px; border: 1px solid #e5e7eb; background: #f9fafb; cursor: pointer; }
      .viewerbtn:hover { background: #f3f4f6; }
      video { width: 100%; max-width: 1200px; border-radius: 10px; background: #000; }
      .imgs { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 10px; }
      .imgs img { width: 360px; max-width: 100%; border-radius: 10px; border: 1px solid #11182722; }
      code { background: #f3f4f6; padding: 2px 5px; border-radius: 6px; }
      select { font-size: 12px; padding: 4px 8px; border-radius: 10px; border: 1px solid #e5e7eb; background: white; }
    </style>
  </head>
  <body>
    <h1>__TITLE__</h1>
    <div class="hint">在该目录下运行 <code>python3 -m http.server 8000</code> 后用浏览器打开本页面即可观看（不要用 file:// 直接打开）。</div>
    __VIEWER_PANEL__
    <div class="grid">
      __CARDS__
    </div>
    <script>
      (function() {
        if (location.protocol === 'file:') {
          const banner = document.createElement('div');
          banner.textContent = '提示：请不要用 file:// 打开。请在该目录运行 `python3 -m http.server 8000`，然后用 http:// 打开本页。';
          banner.style.cssText = 'padding:10px 12px;margin:0 0 12px;border:1px solid #fecaca;background:#fef2f2;color:#991b1b;border-radius:12px;font-size:12px;';
          document.body.insertBefore(banner, document.body.firstChild);
        }
        const scenes = __SCENES__;
        const frame = document.getElementById('viewerFrame');
        const select = document.getElementById('viewerScene');
        const open = document.getElementById('viewerOpen');
        function setScene(scene) {
          if (!scene || !frame) return;
          const url = 'viewer.html?scene=' + encodeURIComponent(scene);
          frame.src = url;
          if (open) open.href = url;
          if (select) select.value = scene;
        }
        if (select && scenes && scenes.length) {
          select.innerHTML = scenes.map(s => `<option value="${s}">${s}</option>`).join('');
          select.value = scenes[0];
          select.addEventListener('change', (e) => setScene(e.target.value));
        }
        document.querySelectorAll('button.viewerbtn[data-scene]').forEach(btn => {
          btn.addEventListener('click', () => {
            const s = btn.getAttribute('data-scene');
            setScene(s);
            if (frame) frame.scrollIntoView({ behavior: 'smooth', block: 'start' });
          });
        });
      })();
    </script>
  </body>
</html>
"""

    safe_title = _html_escape(args.title)
    html = (
        index_template.replace("__TITLE__", safe_title)
        .replace("__VIEWER_PANEL__", viewer_panel)
        .replace("__CARDS__", "".join(card_html(r) for r in runs))
        .replace("__SCENES__", json.dumps(scenes))
    )
    out_path.write_text(html, encoding="utf-8")
    print(f"[OK] Wrote: {out_path}")


if __name__ == "__main__":
    main()
