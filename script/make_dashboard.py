import argparse
import json
import re
from pathlib import Path


def _read_json(path: Path):
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


def _parse_tail_metrics_from_log(log_path: Path):
    if not log_path.exists():
        return None
    text = log_path.read_text(errors="ignore")
    keys = [
        "accuracy",
        "completion",
        "nc1",
        "nc2",
        "acc_med",
        "comp_med",
        "nc1_med",
        "nc2_med",
        "abs_rel",
        "sq_rel",
        "rmse",
        "delta_1_25",
        "delta_1_25_2",
    ]
    out = {}
    for k in keys:
        m = re.search(rf"^{re.escape(k)}\s+([0-9eE+\-\.]+)\s*$", text, flags=re.MULTILINE)
        if m:
            try:
                out[k] = float(m.group(1))
            except Exception:
                pass
    return out or None


def main():
    p = argparse.ArgumentParser("Generate a single-page dashboard for nuScenes/OPV2V results.")
    p.add_argument("--out", type=Path, default=None, help="Defaults to Driv3R/output/dashboard/index.html")
    args = p.parse_args()

    driv3r_root = Path(__file__).resolve().parents[1]
    repo_root = driv3r_root.parent

    out_path = args.out or (driv3r_root / "output" / "dashboard" / "index.html")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    nus_log = repo_root / "logs" / "eval_nuscenes_dynamic_full_depthnone.log"
    nus_metrics = _parse_tail_metrics_from_log(nus_log) or {}

    def entry(title: str, root: Path, note: str = ""):
        summary = _read_json(root / "summary.json") or {}
        return {
            "title": title,
            "root": root,
            "index_rel": str((root / "index.html").relative_to(driv3r_root / "output")),
            "viewer_rel": str((root / "viewer.html").relative_to(driv3r_root / "output")),
            "summary": summary,
            "note": note,
        }

    opv_runs = [
        entry(
            "OPV2V Base · 单车 · 4相机 · 去天空0.30",
            driv3r_root / "output" / "opv2v_coop_eval" / "base_single_allcam_nosky30",
            "nuScenes 官方 ckpt 直接迁移到 OPV2V（未微调）",
        ),
        entry(
            "OPV2V Base · 多车协同(<=5车) · 4相机 · 去天空0.30",
            driv3r_root / "output" / "opv2v_coop_eval" / "base_coop_allcam_nosky30",
            "多车结果 = 多 CAV 点云并集（非特征级融合）",
        ),
        entry(
            "OPV2V 单车微调 egoft_v2 · 单车 · 4相机 · 去天空0.30",
            driv3r_root / "output" / "opv2v_coop_eval" / "egoft_v2_single_allcam_nosky30",
            "从 egoft(2e/2000) 继续训到 4e/8000",
        ),
        entry(
            "OPV2V 单车微调 egoft_v2 · 多车协同(<=5车) · 4相机 · 去天空0.30",
            driv3r_root / "output" / "opv2v_coop_eval" / "egoft_v2_coop_allcam_nosky30",
            "多车结果 = 多 CAV 点云并集（非特征级融合）",
        ),
        entry(
            "OPV2V 多车微调 allft_v2 · 单车 · 4相机 · 去天空0.30",
            driv3r_root / "output" / "opv2v_coop_eval" / "allft_v2_single_allcam_nosky30",
            "从 egoft_v2 继续训：cav_mode=all,max_cav=5",
        ),
        entry(
            "OPV2V 多车微调 allft_v2 · 多车协同(<=5车) · 4相机 · 去天空0.30",
            driv3r_root / "output" / "opv2v_coop_eval" / "allft_v2_coop_allcam_nosky30",
            "从 egoft_v2 继续训：cav_mode=all,max_cav=5",
        ),
    ]

    opv_runs_segsky = [
        entry(
            "OPV2V Base · 单车 · 4相机 · SegFormer sky mask",
            driv3r_root / "output" / "opv2v_coop_eval_segsky" / "base_single_allcam_segsky",
            "语义分割去掉 sky 类别（ADE20K / SegFormer-b0），可视化更干净但可能伤害指标",
        ),
        entry(
            "OPV2V Base · 多车协同(<=5车) · 4相机 · SegFormer sky mask",
            driv3r_root / "output" / "opv2v_coop_eval_segsky" / "base_coop_allcam_segsky",
            "多车结果 = 多 CAV 点云并集（非特征级融合）",
        ),
        entry(
            "OPV2V 单车微调 egoft_v2 · 单车 · 4相机 · SegFormer sky mask",
            driv3r_root / "output" / "opv2v_coop_eval_segsky" / "egoft_v2_single_allcam_segsky",
            "从 egoft(2e/2000) 继续训到 4e/8000",
        ),
        entry(
            "OPV2V 单车微调 egoft_v2 · 多车协同(<=5车) · 4相机 · SegFormer sky mask",
            driv3r_root / "output" / "opv2v_coop_eval_segsky" / "egoft_v2_coop_allcam_segsky",
            "多车结果 = 多 CAV 点云并集（非特征级融合）",
        ),
        entry(
            "OPV2V 多车微调 allft_v2 · 单车 · 4相机 · SegFormer sky mask",
            driv3r_root / "output" / "opv2v_coop_eval_segsky" / "allft_v2_single_allcam_segsky",
            "从 egoft_v2 继续训：cav_mode=all,max_cav=5",
        ),
        entry(
            "OPV2V 多车微调 allft_v2 · 多车协同(<=5车) · 4相机 · SegFormer sky mask",
            driv3r_root / "output" / "opv2v_coop_eval_segsky" / "allft_v2_coop_allcam_segsky",
            "从 egoft_v2 继续训：cav_mode=all,max_cav=5",
        ),
    ]

    opv_best_runs = [
        entry(
            "OPV2V BestEval · Base · 单车 · 4相机 · d[0.5,150] · 去天空0.30",
            driv3r_root / "output" / "opv2v_coop_eval_best" / "base_single_allcam_d0p5_150_nosky30",
            "统一评测过滤：depth∈[0.5,150] + top30% sky drop",
        ),
        entry(
            "OPV2V BestEval · Base · 多车协同(<=5车) · 4相机 · d[0.5,150] · 去天空0.30",
            driv3r_root / "output" / "opv2v_coop_eval_best" / "base_coop_allcam_d0p5_150_nosky30",
            "多车结果 = 多 CAV 点云并集（非特征级融合）",
        ),
        entry(
            "OPV2V BestEval · egoft_v2 · 单车 · 4相机 · d[0.5,150] · 去天空0.30",
            driv3r_root / "output" / "opv2v_coop_eval_best" / "egoft_v2_single_allcam_d0p5_150_nosky30",
            "OPV2V 单车微调 ckpt",
        ),
        entry(
            "OPV2V BestEval · egoft_v2 · 多车协同(<=5车) · 4相机 · d[0.5,150] · 去天空0.30",
            driv3r_root / "output" / "opv2v_coop_eval_best" / "egoft_v2_coop_allcam_d0p5_150_nosky30",
            "多车结果 = 多 CAV 点云并集（非特征级融合）",
        ),
        entry(
            "OPV2V BestEval · allft_v2 · 单车 · 4相机 · d[0.5,150] · 去天空0.30",
            driv3r_root / "output" / "opv2v_coop_eval_best" / "allft_v2_single_allcam_d0p5_150_nosky30",
            "OPV2V 多车微调 ckpt（cav_mode=all,max_cav=5）",
        ),
        entry(
            "OPV2V BestEval · allft_v2 · 多车协同(<=5车) · 4相机 · d[0.5,150] · 去天空0.30",
            driv3r_root / "output" / "opv2v_coop_eval_best" / "allft_v2_coop_allcam_d0p5_150_nosky30",
            "OPV2V 多车微调 ckpt（cav_mode=all,max_cav=5）",
        ),
    ]

    # These are NOT part of the requested 6-run table; they are extra configs that usually
    # look closer to nuScenes visualizations (more coverage with both crops + longer seq).
    opv_vis_runs = [
        entry(
            "OPV2V allft_v2 · 单车 · sky0.40 · allcam · both · seq15",
            driv3r_root
            / "output"
            / "opv2v_coop_eval_best"
            / "allft_v2_single_allcam_d0p5_150_nosky40_both_seq15_mem1p0",
            "推荐可视化配置：sky_top_ratio=0.40, split_view=both, sequence_length=15",
        ),
        entry(
            "OPV2V allft_v2 · 协同(<=5车) · sky0.40 · allcam · both · seq10",
            driv3r_root
            / "output"
            / "opv2v_coop_eval_best"
            / "allft_v2_coop_allcam_d0p5_150_nosky40_both_seq10_mem1p0",
            "推荐可视化配置：sky_top_ratio=0.40, split_view=both, sequence_length=10",
        ),
    ]

    def opv_row_html(e):
        s = e["summary"] or {}
        return f"""
        <tr>
          <td class="name">
            <div class="t">{_html_escape(e['title'])}</div>
            <div class="n">{_html_escape(e.get('note',''))}</div>
          </td>
          <td>{_fmt(s.get('accuracy'))}</td>
          <td>{_fmt(s.get('completion'))}</td>
          <td>{_fmt(s.get('completion_ratio'))}</td>
          <td><a href="../{_html_escape(e['index_rel'])}">open</a></td>
        </tr>
        """.strip()

    def opv_segsky_row_html(e):
        s = e["summary"] or {}
        return f"""
        <tr>
          <td class="name">
            <div class="t">{_html_escape(e['title'])}</div>
            <div class="n">{_html_escape(e.get('note',''))}</div>
          </td>
          <td>{_fmt(s.get('accuracy_raw'))}</td>
          <td>{_fmt(s.get('completion_raw'))}</td>
          <td>{_fmt(s.get('completion_ratio_raw'))}</td>
          <td>{_fmt(s.get('accuracy'))}</td>
          <td>{_fmt(s.get('completion'))}</td>
          <td>{_fmt(s.get('completion_ratio'))}</td>
          <td>{_fmt(s.get('icp_rot_deg'))}</td>
          <td>{_fmt(s.get('icp_centered_t_norm'))}</td>
          <td><a href="../{_html_escape(e['index_rel'])}">open</a></td>
        </tr>
        """.strip()

    nus_vis_links = [
        ("nuScenes 全局点云(去天空0.30, conf>=3) · 3个scene", "nuscenes_vis_conf3_nosky30/index.html"),
        ("nuScenes ICP对齐(含天空) · 3个序列", "vis_icp_gallery/nuscenes_dynamic_baseline_sky0_conf3_d2_80/index.html"),
        ("nuScenes ICP对齐(去天空0.30) · 3个序列", "vis_icp_gallery/nuscenes_dynamic_baseline_nosky30_conf3_d2_80/index.html"),
    ]

    nus_metrics_lines = [
        ("Acc(mean)", _fmt(nus_metrics.get("accuracy"), 4)),
        ("Comp(mean)", _fmt(nus_metrics.get("completion"), 4)),
        ("Acc(med)", _fmt(nus_metrics.get("acc_med"), 4)),
        ("Comp(med)", _fmt(nus_metrics.get("comp_med"), 4)),
        ("AbsRel", _fmt(nus_metrics.get("abs_rel"), 4)),
        ("RMSE", _fmt(nus_metrics.get("rmse"), 4)),
        ("δ<1.25", _fmt(nus_metrics.get("delta_1_25"), 4)),
    ]

    nus_metrics_html = " ".join([f"<span><b>{k}</b>: {v}</span>" for k, v in nus_metrics_lines])
    nus_links_html = "".join([f'<li><a href="../{href}">{_html_escape(name)}</a></li>' for name, href in nus_vis_links])

    viewer_runs = [
        {"title": e["title"], "viewer_rel": e["viewer_rel"], "index_rel": e["index_rel"]}
        for e in (opv_best_runs + opv_vis_runs + opv_runs + opv_runs_segsky)
    ]
    viewer_runs_json = json.dumps(viewer_runs, ensure_ascii=False)

    html = f"""<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Driv3R 复现与 OPV2V 适配 - Dashboard</title>
    <style>
      body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Noto Sans", "Liberation Sans"; margin: 24px; }}
      h1 {{ margin: 0 0 10px; }}
      h2 {{ margin: 22px 0 10px; }}
      .hint {{ color: #6b7280; margin: 8px 0 18px; font-size: 12px; }}
      code {{ background: #f3f4f6; padding: 2px 5px; border-radius: 6px; }}
      .metrics {{ display: flex; flex-wrap: wrap; gap: 10px; font-size: 12px; color: #111827; margin: 8px 0 10px; }}
      .card {{ padding: 14px; border: 1px solid #e5e7eb; border-radius: 12px; }}
      ul {{ margin: 8px 0 0 18px; }}
      a {{ color: #2563eb; text-decoration: none; }}
      a:hover {{ text-decoration: underline; }}
      table {{ width: 100%; border-collapse: collapse; }}
      th, td {{ border-bottom: 1px solid #e5e7eb; padding: 10px 8px; font-size: 12px; text-align: left; vertical-align: top; }}
      th {{ background: #f9fafb; font-weight: 700; }}
      td.name .t {{ font-weight: 700; }}
      td.name .n {{ color: #6b7280; margin-top: 4px; }}
      .viewer-card {{ display: grid; grid-template-rows: auto 1fr; gap: 10px; }}
      .viewer-toolbar {{ display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }}
      .viewer-toolbar select {{ max-width: min(920px, 100%); }}
      iframe.viewer {{ width: 100%; height: min(78vh, 840px); border: 1px solid #e5e7eb; border-radius: 12px; }}
    </style>
  </head>
  <body>
    <h1>Driv3R 复现与 OPV2V 适配 - Dashboard</h1>
    <div class="hint">在 <code>Driv3R/output</code> 目录运行 <code>python3 -m http.server 8000</code>，然后用浏览器打开本页面（不要用 file://）。</div>

    <h2>nuScenes 基准</h2>
    <div class="card">
      <div><b>Eval(日志)</b>: <code>{_html_escape(str(nus_log))}</code></div>
      <div class="metrics">{nus_metrics_html}</div>
      <div class="hint">注：论文 Table 2 的 Acc/Comp/median 与本次评测基本一致（我们用的动态子集为 dynamic_metas.pkl，共 2597 序列）。</div>
      <div><b>可视化</b>:</div>
      <ul>
        {nus_links_html}
      </ul>
    </div>

	    <h2>OPV2V 六组结果（9个固定场景子集）· BestEval（d[0.5,150] + 去天空0.30）</h2>
    <div class="card">
      <table>
        <thead>
          <tr>
            <th>Run</th>
            <th>Acc ↓</th>
            <th>Comp ↓</th>
            <th>CR@0.2 ↑</th>
            <th>Gallery</th>
          </tr>
        </thead>
        <tbody>
          {"".join(opv_row_html(e) for e in opv_best_runs)}
        </tbody>
      </table>
	    </div>

	    <h2>OPV2V 推荐可视化配置（更像 nuScenes：both + longer seq）</h2>
	    <div class="card">
	      <div class="hint">这部分用于辅助观察可视化趋势：更多视角/更长序列通常 completion 更好，但可能会引入更多漂浮点导致 accuracy 变差。</div>
	      <table>
	        <thead>
	          <tr>
	            <th>Run</th>
	            <th>Acc ↓</th>
	            <th>Comp ↓</th>
	            <th>CR@0.2 ↑</th>
	            <th>Gallery</th>
	          </tr>
	        </thead>
	        <tbody>
	          {"".join(opv_row_html(e) for e in opv_vis_runs)}
	        </tbody>
	      </table>
	    </div>

    <h2>OPV2V 六组结果（9个固定场景子集）· 去天空0.30（旧配置对照）</h2>
    <div class="card">
      <table>
        <thead>
          <tr>
            <th>Run</th>
            <th>Acc ↓</th>
            <th>Comp ↓</th>
            <th>CR@0.2 ↑</th>
            <th>Gallery</th>
          </tr>
        </thead>
        <tbody>
          {"".join(opv_row_html(e) for e in opv_runs)}
        </tbody>
      </table>
    </div>

    <h2>一页内快速预览（交互式点云）</h2>
    <div class="card viewer-card">
      <div class="viewer-toolbar">
        <span><b>选择 Run</b>:</span>
        <select id="run-select"></select>
        <a id="open-run" href="#" target="_blank">open in new tab</a>
        <span class="hint">提示：点云较大时可在 viewer 内调小 point size / 选择 *_web.ply（默认开启 lite）。</span>
      </div>
      <iframe class="viewer" id="viewer-frame" src=""></iframe>
    </div>

    <h2>OPV2V 六组结果（9个固定场景子集）· SegFormer sky mask</h2>
    <div class="card">
      <div class="hint">该表同时展示 raw(不做ICP) 与 icp(做ICP) 指标，用于辅助判断坐标系/对齐问题。</div>
      <table>
        <thead>
          <tr>
            <th>Run</th>
            <th>Acc raw ↓</th>
            <th>Comp raw ↓</th>
            <th>CR raw@0.2 ↑</th>
            <th>Acc icp ↓</th>
            <th>Comp icp ↓</th>
            <th>CR icp@0.2 ↑</th>
            <th>ICP rot(deg)</th>
            <th>ICP t(center)</th>
            <th>Gallery</th>
          </tr>
        </thead>
        <tbody>
          {"".join(opv_segsky_row_html(e) for e in opv_runs_segsky)}
        </tbody>
      </table>
    </div>

    <script>
      const runs = {viewer_runs_json};

      const sel = document.getElementById('run-select');
      const frame = document.getElementById('viewer-frame');
      const open = document.getElementById('open-run');

      function setRun(idx) {{
        const r = runs[idx] || runs[0];
        if (!r) return;
        const viewer = '../' + r.viewer_rel;
        const index = '../' + r.index_rel;
        frame.src = viewer;
        open.href = index;
      }}

      runs.forEach((r, i) => {{
        const opt = document.createElement('option');
        opt.value = String(i);
        opt.textContent = r.title;
        sel.appendChild(opt);
      }});

      sel.addEventListener('change', (e) => setRun(parseInt(e.target.value || '0', 10) || 0));
      setRun(0);
    </script>
  </body>
</html>
"""

    out_path.write_text(html, encoding="utf-8")
    print(f"[OK] Wrote: {out_path}")


if __name__ == "__main__":
    main()
