"""Human verification tool for SAM-proposed defect masks.

Phase 1 of this project requires that every defect mask used as ground truth has
been seen and accepted by a person. This is the tool that makes that practical
rather than merely stated: proposals arrive pre-computed, the reviewer's job is
to pick the right candidate, fix it with a brush if needed, or say there is no
visible defect. One keystroke per kernel in the common case.

It runs standalone on its own port and shares nothing with backend/main.py, so a
review session cannot disturb the application. It writes only to the proposals
CSV and to final_mask.png files inside the proposal directories; the source
images are opened read-only and never modified.

    python -m src.annotation.review_server
    -> http://127.0.0.1:8011

Decisions recorded per kernel:
    accepted        a candidate was correct as proposed
    corrected       a candidate was edited with the brush before accepting
    drawn           no candidate was usable; the reviewer painted the mask
    no_defect       nothing visibly defective (the mask is legitimately empty)
    rejected        unusable image, or the body mask is wrong -- excluded entirely
Only accepted / corrected / drawn / no_defect rows become training data.
"""
from __future__ import annotations

import argparse
import base64
import csv
import datetime
import io
import json
import os

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from src.annotation.sam_propose import PROPOSAL_CSV

VALID_DECISIONS = {"accepted", "corrected", "drawn", "no_defect", "rejected"}

app = FastAPI(title="Maize defect mask review", docs_url=None, redoc_url=None)
STATE: dict = {"csv_path": PROPOSAL_CSV, "rows": [], "fields": []}


def _load() -> None:
    with open(STATE["csv_path"], newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        STATE["rows"] = list(reader)
        STATE["fields"] = list(reader.fieldnames or [])


def _save() -> None:
    # Rewritten in full on every decision. The file is a few hundred KB and a
    # review session is long; a crash twenty kernels in should not cost the
    # nineteen that were already decided.
    tmp = STATE["csv_path"] + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=STATE["fields"])
        writer.writeheader()
        writer.writerows(STATE["rows"])
    os.replace(tmp, STATE["csv_path"])


def _png_data_uri(image: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise HTTPException(500, "failed to encode png")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


@app.get("/api/progress")
def progress() -> dict:
    counts: dict[str, int] = {}
    for row in STATE["rows"]:
        key = row["decision"] or "pending"
        counts[key] = counts.get(key, 0) + 1
    total = len(STATE["rows"])
    done = sum(v for k, v in counts.items() if k != "pending")
    return {"total": total, "reviewed": done, "counts": counts}


@app.get("/api/first_pending")
def first_pending() -> dict:
    """Where to resume. Review runs over several sittings; restarting at row 0
    every time would make the reviewer scroll past their own finished work."""
    for i, row in enumerate(STATE["rows"]):
        if not row["decision"]:
            return {"index": i}
    return {"index": max(len(STATE["rows"]) - 1, 0)}


@app.get("/api/item/{index}")
def item(index: int) -> dict:
    if not 0 <= index < len(STATE["rows"]):
        raise HTTPException(404, "index out of range")
    row = STATE["rows"][index]
    bgr = cv2.imread(row["image_path"])
    if bgr is None:
        raise HTTPException(404, f"unreadable image {row['image_path']}")
    body = cv2.imread(os.path.join(row["proposal_dir"], row["body_mask"]), 0)

    candidates = []
    for cand in json.loads(row["candidates_json"] or "[]"):
        mask = cv2.imread(os.path.join(row["proposal_dir"], cand["file"]), 0)
        if mask is None:
            continue
        candidates.append({"file": cand["file"], "source": cand["source"],
                           "coverage_pct": cand["coverage_pct"],
                           "mask": _png_data_uri(mask)})

    existing = None
    if row.get("final_mask"):
        prev = cv2.imread(os.path.join(row["proposal_dir"], row["final_mask"]), 0)
        if prev is not None:
            existing = _png_data_uri(prev)

    return {
        "index": index,
        "image": _png_data_uri(bgr),
        "body": _png_data_uri(body) if body is not None else None,
        "body_plausible": row["body_plausible"] == "1",
        "grainspace_class": row["grainspace_class"],
        "defect_channel": row["defect_channel"],
        "split": row["split"],
        "group": row["group"],
        "width": int(bgr.shape[1]),
        "height": int(bgr.shape[0]),
        "candidates": candidates,
        "decision": row["decision"],
        "review_note": row.get("review_note", ""),
        "final_mask": existing,
    }


class Decision(BaseModel):
    index: int
    decision: str
    reviewer: str = ""
    note: str = ""
    # PNG data URI of the final mask, white = defect. Omitted for no_defect and
    # rejected, where there is nothing to store.
    mask_png: str | None = None


@app.post("/api/decide")
def decide(payload: Decision) -> JSONResponse:
    if payload.decision not in VALID_DECISIONS:
        raise HTTPException(400, f"decision must be one of {sorted(VALID_DECISIONS)}")
    if not 0 <= payload.index < len(STATE["rows"]):
        raise HTTPException(404, "index out of range")
    row = STATE["rows"][payload.index]

    final_name = ""
    if payload.decision in {"accepted", "corrected", "drawn"}:
        if not payload.mask_png:
            raise HTTPException(400, f"decision '{payload.decision}' requires a mask")
        raw = base64.b64decode(payload.mask_png.split(",", 1)[-1])
        mask = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise HTTPException(400, "could not decode mask png")
        if mask.ndim == 3:
            # The browser canvas paints RGBA; alpha is what carries the strokes.
            mask = mask[:, :, 3] if mask.shape[2] == 4 else cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = (mask > 127).astype(np.uint8) * 255
        # A mask is only meaningful at the resolution of the image it describes.
        # The canvas is kept at native size so this should never fire, but a
        # silently rescaled mask would silently rescale every defect area
        # computed from it, so the guard stays.
        src = cv2.imread(row["image_path"])
        if src is not None and mask.shape[:2] != src.shape[:2]:
            mask = cv2.resize(mask, (src.shape[1], src.shape[0]), interpolation=cv2.INTER_NEAREST)
        final_name = "final_mask.png"
        cv2.imwrite(os.path.join(row["proposal_dir"], final_name), mask)

    row["decision"] = payload.decision
    row["status"] = "rejected" if payload.decision == "rejected" else "verified"
    row["reviewer"] = payload.reviewer or row.get("reviewer") or "human"
    row["reviewed_at"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    row["final_mask"] = final_name
    row["review_note"] = payload.note
    _save()
    return JSONResponse({"ok": True, "progress": progress()})


@app.get("/", response_class=HTMLResponse)
def index_page() -> str:
    return _PAGE


_PAGE = r"""
<!doctype html><html><head><meta charset="utf-8"><title>Defect mask review</title>
<style>
:root{--bg:#14161a;--fg:#e8eaed;--mut:#9aa0a6;--acc:#4fc3f7;--ok:#66bb6a;--warn:#ffa726;--bad:#ef5350}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 ui-sans-serif,system-ui,Segoe UI,sans-serif}
header{display:flex;gap:16px;align-items:center;padding:10px 16px;border-bottom:1px solid #2a2e35;flex-wrap:wrap}
h1{font-size:15px;margin:0;font-weight:600}
.pill{background:#22262d;border-radius:999px;padding:3px 10px;font-size:12px;color:var(--mut)}
main{display:flex;gap:18px;padding:16px;align-items:flex-start;flex-wrap:wrap}
#stage{position:relative;border:1px solid #2a2e35;border-radius:8px;overflow:hidden;background:#0d0f12}
#stage canvas{display:block;position:absolute;top:0;left:0}
#base{position:relative}
.cands{display:flex;gap:8px;flex-wrap:wrap;max-width:640px}
.cand{border:2px solid #2a2e35;border-radius:6px;padding:4px;cursor:pointer;background:#1a1d22;width:118px}
.cand.sel{border-color:var(--acc)}
.cand canvas{display:block;border-radius:3px;width:108px;height:108px}
.cand .lbl{font-size:10px;color:var(--mut);margin-top:3px;text-align:center;white-space:nowrap;overflow:hidden}
aside{min-width:260px;max-width:320px}
kbd{background:#22262d;border:1px solid #343941;border-radius:4px;padding:1px 6px;font-size:11px}
table{border-collapse:collapse;width:100%;font-size:12px}
td{padding:3px 4px;border-bottom:1px solid #22262d;color:var(--mut)}
td:last-child{color:var(--fg);text-align:right}
.row{display:flex;gap:8px;align-items:center;margin:8px 0;flex-wrap:wrap}
button{background:#22262d;color:var(--fg);border:1px solid #343941;border-radius:6px;padding:6px 11px;cursor:pointer;font-size:13px}
button:hover{border-color:var(--acc)}
button.primary{background:var(--acc);color:#0d0f12;border-color:var(--acc);font-weight:600}
input[type=text]{background:#1a1d22;border:1px solid #343941;color:var(--fg);border-radius:6px;padding:5px 8px;width:100%}
.bar{height:5px;background:#22262d;border-radius:3px;overflow:hidden;min-width:150px;flex:1}
.bar>div{height:100%;background:var(--ok);width:0}
.warn{color:var(--warn)} .mut{color:var(--mut)} .bad{color:var(--bad)}
</style></head><body>
<header>
  <h1>Defect mask review</h1>
  <span class="pill" id="pos">-</span>
  <span class="pill" id="cls">-</span>
  <span class="pill" id="split">-</span>
  <div class="bar"><div id="prog"></div></div>
  <span class="pill" id="stats">-</span>
</header>
<main>
  <div>
    <div id="stage"></div>
    <div class="row">
      <button onclick="prev()">&larr; Prev <kbd>&larr;</kbd></button>
      <button onclick="next()">Next &rarr; <kbd>&rarr;</kbd></button>
      <span class="mut">brush <kbd>[</kbd><kbd>]</kbd> size &middot; <kbd>E</kbd> erase &middot; <kbd>Z</kbd> undo &middot; <kbd>C</kbd> clear</span>
    </div>
    <div class="cands" id="cands"></div>
  </div>
  <aside>
    <table>
      <tr><td>channel</td><td id="chan">-</td></tr>
      <tr><td>plate group</td><td id="grp" style="font-size:10px">-</td></tr>
      <tr><td>body mask</td><td id="bodyok">-</td></tr>
      <tr><td>coverage of body</td><td id="cov">-</td></tr>
      <tr><td>current decision</td><td id="dec">-</td></tr>
    </table>
    <div class="row"><input type="text" id="reviewer" placeholder="reviewer name"></div>
    <div class="row"><input type="text" id="note" placeholder="note (optional)"></div>
    <div class="row">
      <button class="primary" onclick="commit('auto')">Accept <kbd>Enter</kbd></button>
      <button onclick="commit('no_defect')">No defect <kbd>0</kbd></button>
      <button onclick="commit('rejected')">Reject <kbd>X</kbd></button>
    </div>
    <p class="mut" style="font-size:12px">
      <kbd>1</kbd>&ndash;<kbd>9</kbd> pick a candidate. Painting after picking records the
      decision as <b>corrected</b>; painting from an empty mask records <b>drawn</b>.
      <b>Reject</b> drops the kernel from the dataset entirely &mdash; use it when the body
      mask is wrong or the crop is unusable.
    </p>
  </aside>
</main>
<script>
let idx=0, TOTAL=0, data=null, sel=-1, painted=false, brush=9, erasing=false, undo=[];
const stage=document.getElementById('stage');
let img=new Image(), baseC, maskC, bctx, mctx, scale=1;

async function load(i){
  const r=await fetch('/api/item/'+i); if(!r.ok){return;}
  data=await r.json(); idx=data.index; sel=-1; painted=false; undo=[];
  document.getElementById('pos').textContent=(idx+1)+' / '+(TOTAL||'?');
  document.getElementById('cls').textContent=data.grainspace_class;
  document.getElementById('split').textContent=data.split;
  document.getElementById('chan').textContent=data.defect_channel||'(none)';
  document.getElementById('grp').textContent=data.group;
  document.getElementById('bodyok').innerHTML=data.body_plausible?'<span style="color:#66bb6a">ok</span>':'<span class="bad">implausible</span>';
  document.getElementById('dec').textContent=data.decision||'pending';
  document.getElementById('note').value=data.review_note||'';
  await setupCanvas();
  renderCands();
  if(data.final_mask){ await paintFrom(data.final_mask); }
  updateCov();
}

function fit(w,h){ const m=520; scale=Math.min(m/w,m/h,3); return [Math.round(w*scale),Math.round(h*scale)]; }

async function setupCanvas(){
  await new Promise(res=>{img.onload=res; img.src=data.image;});
  const [w,h]=fit(data.width,data.height);
  stage.style.width=w+'px'; stage.style.height=h+'px';
  stage.innerHTML='';
  baseC=document.createElement('canvas'); baseC.width=w; baseC.height=h;
  maskC=document.createElement('canvas'); maskC.width=data.width; maskC.height=data.height;
  maskC.style.width=w+'px'; maskC.style.height=h+'px'; maskC.style.opacity=0.55;
  stage.appendChild(baseC); stage.appendChild(maskC);
  bctx=baseC.getContext('2d'); mctx=maskC.getContext('2d');
  bctx.drawImage(img,0,0,w,h);
  if(data.body) await drawBodyOutline();
  mctx.clearRect(0,0,maskC.width,maskC.height);
  attachPaint();
}

// The body mask is the DENOMINATOR of every coverage % this project reports, so
// the reviewer has to be able to see it. Drawn as an outline rather than a fill
// so it never hides the defect it is measuring.
async function drawBodyOutline(){
  const b=new Image(); await new Promise(r=>{b.onload=r; b.onerror=r; b.src=data.body;});
  const t=document.createElement('canvas'); t.width=data.width; t.height=data.height;
  const tc=t.getContext('2d'); tc.drawImage(b,0,0);
  const px=tc.getImageData(0,0,data.width,data.height), d=px.data, W=data.width;
  const out=tc.createImageData(data.width,data.height);
  for(let y=1;y<data.height-1;y++) for(let x=1;x<W-1;x++){
    const i=(y*W+x)*4;
    if(d[i]<=127) continue;
    const edge = d[i-4]<=127 || d[i+4]<=127 || d[i-W*4]<=127 || d[i+W*4]<=127;
    if(edge){ out.data[i]=79; out.data[i+1]=195; out.data[i+2]=247; out.data[i+3]=255; }
  }
  tc.putImageData(out,0,0);
  bctx.drawImage(t,0,0,baseC.width,baseC.height);
}

function renderCands(){
  const box=document.getElementById('cands'); box.innerHTML='';
  (data.candidates||[]).forEach((c,i)=>{
    const d=document.createElement('div'); d.className='cand'; d.onclick=()=>pick(i);
    const cv=document.createElement('canvas'); cv.width=108; cv.height=108;
    const cx=cv.getContext('2d'); cx.drawImage(img,0,0,108,108);
    const m=new Image();
    // The stored masks are white-on-black. Drawn as-is at partial alpha a
    // candidate would merely brighten the crop and be near-invisible against a
    // pale kernel, so each is recoloured to transparent-red first.
    m.onload=()=>{
      const t=document.createElement('canvas'); t.width=data.width; t.height=data.height;
      const tc=t.getContext('2d'); tc.drawImage(m,0,0,data.width,data.height);
      const px=tc.getImageData(0,0,data.width,data.height);
      for(let k=0;k<px.data.length;k+=4){
        const on=px.data[k]>127;
        px.data[k]=239; px.data[k+1]=83; px.data[k+2]=80; px.data[k+3]=on?165:0;
      }
      tc.putImageData(px,0,0); cx.drawImage(t,0,0,108,108);
    };
    m.src=c.mask;
    const l=document.createElement('div'); l.className='lbl';
    l.textContent=(i+1)+' · '+c.source+' · '+c.coverage_pct.toFixed(1)+'%';
    d.appendChild(cv); d.appendChild(l); d.id='cand'+i; box.appendChild(d);
  });
}

async function paintFrom(src){
  const m=new Image(); await new Promise(r=>{m.onload=r; m.src=src;});
  const t=document.createElement('canvas'); t.width=data.width; t.height=data.height;
  const tc=t.getContext('2d'); tc.drawImage(m,0,0,data.width,data.height);
  const px=tc.getImageData(0,0,data.width,data.height);
  const d=px.data;
  for(let i=0;i<d.length;i+=4){ const on=d[i]>127; d[i]=239; d[i+1]=83; d[i+2]=80; d[i+3]=on?255:0; }
  mctx.putImageData(px,0,0);
}

async function pick(i){
  sel=i; painted=false; undo=[];
  document.querySelectorAll('.cand').forEach(e=>e.classList.remove('sel'));
  const el=document.getElementById('cand'+i); if(el) el.classList.add('sel');
  mctx.clearRect(0,0,maskC.width,maskC.height);
  await paintFrom(data.candidates[i].mask);
  updateCov();
}

function attachPaint(){
  let down=false;
  const pos=e=>{const r=maskC.getBoundingClientRect();
    return [(e.clientX-r.left)/r.width*maskC.width,(e.clientY-r.top)/r.height*maskC.height];};
  const dot=(x,y)=>{ mctx.globalCompositeOperation = erasing?'destination-out':'source-over';
    mctx.fillStyle='rgba(239,83,80,1)'; mctx.beginPath();
    mctx.arc(x,y,brush,0,7); mctx.fill(); mctx.globalCompositeOperation='source-over'; };
  maskC.addEventListener('pointerdown',e=>{down=true;
    undo.push(mctx.getImageData(0,0,maskC.width,maskC.height)); if(undo.length>25)undo.shift();
    painted=true; const [x,y]=pos(e); dot(x,y); maskC.setPointerCapture(e.pointerId); updateCov();});
  maskC.addEventListener('pointermove',e=>{ if(!down)return; const [x,y]=pos(e); dot(x,y); });
  maskC.addEventListener('pointerup',()=>{down=false; updateCov();});
  maskC.style.cursor='crosshair';
}

function maskPixels(){
  const d=mctx.getImageData(0,0,maskC.width,maskC.height).data; let n=0;
  for(let i=3;i<d.length;i+=4) if(d[i]>127) n++;
  return n;
}
function updateCov(){
  const n=maskPixels();
  document.getElementById('cov').textContent=n?n+' px':'empty';
}

function exportMask(){
  const t=document.createElement('canvas'); t.width=maskC.width; t.height=maskC.height;
  const tc=t.getContext('2d');
  const src=mctx.getImageData(0,0,maskC.width,maskC.height);
  const out=tc.createImageData(maskC.width,maskC.height);
  for(let i=0;i<src.data.length;i+=4){
    const on=src.data[i+3]>127?255:0;
    out.data[i]=out.data[i+1]=out.data[i+2]=on; out.data[i+3]=255;
  }
  tc.putImageData(out,0,0);
  return t.toDataURL('image/png');
}

async function commit(kind){
  if(!data) return;
  let decision=kind, mask=null;
  if(kind==='auto'){
    const n=maskPixels();
    if(n===0){ decision='no_defect'; }
    else if(sel>=0) decision = painted?'corrected':'accepted';
    else decision='drawn';
  }
  if(['accepted','corrected','drawn'].includes(decision)) mask=exportMask();
  const r=await fetch('/api/decide',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({index:idx,decision:decision,mask_png:mask,
      reviewer:document.getElementById('reviewer').value,
      note:document.getElementById('note').value})});
  if(!r.ok){ alert('save failed: '+await r.text()); return; }
  const j=await r.json(); showProgress(j.progress); next();
}

function showProgress(p){
  TOTAL=p.total;
  document.getElementById('prog').style.width=(p.reviewed/p.total*100)+'%';
  document.getElementById('stats').textContent=p.reviewed+' / '+p.total+' · '+
    Object.entries(p.counts).filter(([k])=>k!=='pending').map(([k,v])=>k+' '+v).join('  ');
}
function next(){ load(idx+1); } function prev(){ if(idx>0) load(idx-1); }

document.addEventListener('keydown',e=>{
  if(e.target.tagName==='INPUT') return;
  if(e.key>='1'&&e.key<='9'){ pick(parseInt(e.key)-1); e.preventDefault(); }
  else if(e.key==='0'){ commit('no_defect'); }
  else if(e.key==='Enter'){ commit('auto'); }
  else if(e.key.toLowerCase()==='x'){ commit('rejected'); }
  else if(e.key==='ArrowRight'){ next(); } else if(e.key==='ArrowLeft'){ prev(); }
  else if(e.key.toLowerCase()==='e'){ erasing=!erasing; }
  else if(e.key==='['){ brush=Math.max(1,brush-2); } else if(e.key===']'){ brush=Math.min(60,brush+2); }
  else if(e.key.toLowerCase()==='c'){ mctx.clearRect(0,0,maskC.width,maskC.height); painted=true; updateCov(); }
  else if(e.key.toLowerCase()==='z'){ const u=undo.pop(); if(u){ mctx.putImageData(u,0,0); updateCov(); } }
});

(async()=>{
  showProgress(await (await fetch('/api/progress')).json());
  const start=await (await fetch('/api/first_pending')).json();
  await load(start.index);
})();
</script></body></html>
"""


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=PROPOSAL_CSV)
    ap.add_argument("--port", type=int, default=8011)
    args = ap.parse_args()

    STATE["csv_path"] = args.csv
    _load()
    if not STATE["rows"]:
        raise SystemExit(f"{args.csv} has no rows. Run src/annotation/sam_propose.py first.")
    pending = sum(1 for r in STATE["rows"] if not r["decision"])
    print(f"{len(STATE['rows'])} proposals loaded, {pending} pending review")
    print(f"open http://127.0.0.1:{args.port}")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
