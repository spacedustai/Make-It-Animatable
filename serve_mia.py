#!/usr/bin/env python3
"""
serve_mia.py  –  GPU Cloud-Run micro-service for Make-It-Animatable
──────────────────────────────────────────────────────────────────
POST /rig  {input_uri, animation_uri?, config{…}}  ➜  gs://…/rigged.glb

• Downloads input & optional animation from GCS
• Calls Make-It-Animatable (inference) **without spawning a subprocess**
• Uploads rigged GLB back to GCS
• Returns JSON {job_id, result_uri}

Dependencies (add to Docker image):
    fastapi uvicorn[standard] google-cloud-storage
    # plus the ordinary requirements-demo.txt from Make-It-Animatable
"""

from __future__ import annotations
import os, json, uuid, tempfile, shutil, pathlib, typing as t

# ── Google Cloud ────────────────────────────────────────────────────────────
from google.cloud import storage

# ── FastAPI ─────────────────────────────────────────────────────────────────
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

# ── Make-It-Animatable internals (monkey-patch Gradio bits) ────────────────
import sys, types, importlib, contextlib

# add repo root to sys.path for `import app`
ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# patch dummy gradio UI symbols so `app.py` imports work headless
app_mod = importlib.import_module("app")
app_mod.state = "state"
for name in [
    "output_joints_coarse", "output_normed_input", "output_sample",
    "output_joints", "output_bw", "output_rest_lbs", "output_rest_vis",
    "output_anim", "output_anim_vis"
]:
    setattr(app_mod, name, name)

from app import (  # type: ignore  pylint: disable=wrong-import-position
    DB, init_models, prepare_input, preprocess, infer, vis, vis_blender,
    clear
)

# ── ENV / CONSTS ────────────────────────────────────────────────────────────
UPLOAD_DIR = pathlib.Path("/tmp")        # Cloud-Run tmpfs
OUTPUT_BUCKET = os.getenv("OUTPUT_BUCKET", "mia-results")

if os.getenv("DISABLE_GCS") == "1":
    gcs = None                       # skip GCS when running locally
else:
    gcs = storage.Client()

# ── Helpers ─────────────────────────────────────────────────────────────────
def split_gs(gs_uri: str) -> tuple[str, str]:
    if not gs_uri.startswith("gs://"):
        raise ValueError("uri must start with gs://")
    bucket, *blob = gs_uri[5:].split("/", 1)
    return bucket, blob[0]

def gcs_download(gs_uri: str, dst: pathlib.Path):
    bkt, blob = split_gs(gs_uri)
    gcs.bucket(bkt).blob(blob).download_to_filename(dst)

def gcs_upload(src: pathlib.Path, gs_uri: str):
    bkt, blob = split_gs(gs_uri)
    gcs.bucket(bkt).blob(blob).upload_from_filename(src)

def extract_db(result: dict) -> "DB":        # result dict stores state key
    return result[app_mod.state]

# ── Core rig function  (no subprocess) ──────────────────────────────────────
DEFAULT_OUTDIR = UPLOAD_DIR / "out"
DEFAULT_BW_VIS_BONE = "LeftArm"

# keys mirror the original CLI flags
class RigConfig(BaseModel):
    # input flags
    is_gs: bool = False
    opacity_threshold: float = 0.01
    no_fingers: bool = False
    rest_pose: str = "No"                    # T-pose / A-pose / 大-pose / No
    rest_parts: list[str] | None = None

    # weight / inference flags
    use_normal: bool = False
    bw_fix: bool = True
    bw_vis_bone: str = DEFAULT_BW_VIS_BONE

    # animation flags
    reset_to_rest: bool = True
    retarget: bool = True
    inplace: bool = True

    # misc
    output_dir: str | None = None

# GCS-based wrapper
def run_mia_service(
    input_uri: str,
    cfg: RigConfig,
    animation_uri: str | None = None,
) -> pathlib.Path:
    """Runs Make-It-Animatable end-to-end and returns local GLB path."""
    init_models()     # idempotent; cached after first call

    with tempfile.TemporaryDirectory(dir=UPLOAD_DIR) as tmpd:
        tmp = pathlib.Path(tmpd)
        local_in = tmp / pathlib.Path(input_uri).name
        gcs_download(input_uri, local_in)

        local_anim: pathlib.Path | None = None
        if animation_uri:
            if animation_uri.startswith("gs://"):
                local_anim = tmp / pathlib.Path(animation_uri).name
                gcs_download(animation_uri, local_anim)
            else:
                # Treat as local file path
                local_anim = pathlib.Path(animation_uri)
                if not local_anim.exists():
                    raise ValueError(f"Local animation file not found: {animation_uri}")

        # choose output dir
        out_dir = pathlib.Path(cfg.output_dir) if cfg.output_dir else tmp / "out"
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1 Prepare input
        db = DB()
        res = prepare_input(
            input_path=str(local_in),
            is_gs=cfg.is_gs,
            opacity_threshold=cfg.opacity_threshold,
            db=db,
            export_temp=False,
        )
        db = extract_db(res)
        db.output_dir = str(out_dir)           # force output dir

        # Override default output paths when caller sets --output-dir
        if cfg.output_dir:
            def fp(name: str): return str(out_dir / name)
            ext_in = local_in.suffix
            db.joints_coarse_path = fp("joints_coarse.glb")
            db.normed_path        = fp(f"normed{ext_in}")
            db.sample_path        = fp("sample.glb")
            db.bw_path            = fp("bw.glb")
            db.joints_path        = fp("joints.glb")
            ext_rest = "ply" if cfg.is_gs else "glb"
            db.rest_lbs_path      = fp(f"rest_lbs.{ext_rest}")
            db.rest_vis_path      = fp("rest.glb")
            input_stem = local_in.stem
            db.anim_path          = fp(f"{input_stem}.{'blend' if cfg.is_gs else 'fbx'}")
            db.anim_vis_path      = fp(f"{input_stem}.glb")

        # 2 Preprocess & 3 Infer
        db = extract_db(preprocess(db))
        db = extract_db(infer(cfg.use_normal, db))

        # 4 Visualise weights
        db = extract_db(
            vis(cfg.bw_fix, cfg.bw_vis_bone, cfg.no_fingers, db)
        )

        # 5 Generate animation (+ retarget)
        # make rest_parts safe
        safe_parts = cfg.rest_parts or []

        # resolve animation path
        anim_path = str(local_anim) if local_anim else None
        
        print("Running Blender step to generate animation")
        db = extract_db(
            vis_blender(
                cfg.reset_to_rest,
                cfg.no_fingers,
                cfg.rest_pose,
                safe_parts,
                anim_path,
                cfg.retarget,
                cfg.inplace,
                db,
            )
        )
        # final GLB to return = db.anim_vis_path
        rigged_glb = pathlib.Path(db.anim_vis_path)


        if not rigged_glb.exists():
            raise RuntimeError("Make-It-Animatable did not produce an output GLB")

        # Upload to GCS before temp dir is deleted
        result_uri = f"gs://{OUTPUT_BUCKET}/rigs/{rigged_glb.name}"
        gcs_upload(rigged_glb, result_uri)

        clear(db)
        return result_uri
        

# ── FastAPI service layer ───────────────────────────────────────────────────
class RigRequest(BaseModel):
    input_uri: str
    animation_uri: str | None = None
    config: RigConfig = Field(default_factory=RigConfig)

class RigResponse(BaseModel):
    job_id: str
    result_uri: str

app = FastAPI(title="Make-It-Animatable-Service", version="2.0-inproc")

@app.get("/healthz")
def health():
    return {"status": "ok"}

@app.post("/rig", response_model=RigResponse)
def rig(req: RigRequest):
    job_id = str(uuid.uuid4())
    try:
        result_uri = run_mia_service(
            input_uri=req.input_uri,
            cfg=req.config,
            animation_uri=req.animation_uri,
        )
        return RigResponse(job_id=job_id, result_uri=result_uri)

    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc

# ── Entrypoint for Docker / Cloud Run ───────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run("serve_mia:app", host="0.0.0.0", port=port, workers=1)
