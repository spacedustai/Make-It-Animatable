#!/usr/bin/env python3
import argparse
import os
import sys
import torch
import tempfile
import gc
from pathlib import Path

# Add the current directory to the path to import from app.py
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Monkey patch the app module to define the state variable
import types
import app

# Create a state variable in the app module
app.state = "state"

# Create mock Gradio UI components
app.output_joints_coarse = "output_joints_coarse"
app.output_normed_input = "output_normed_input"
app.output_sample = "output_sample"
app.output_joints = "output_joints"
app.output_bw = "output_bw"
app.output_rest_lbs = "output_rest_lbs"
app.output_rest_vis = "output_rest_vis"
app.output_anim = "output_anim"
app.output_anim_vis = "output_anim_vis"

# Now import functions from app
from app import (
    DB, init_models, prepare_input, preprocess, infer, vis, vis_blender, finish,
    clear, get_pose_ignore_list, is_main_thread, load_gs, 
    Transform3d, str2bool
)

def parse_args():
    parser = argparse.ArgumentParser(description="Make-It-Animatable: Command Line Interface for rigging and animating 3D models")
    
    # Input model
    parser.add_argument("--input", type=str, required=True, help="Path to input 3D model")
    parser.add_argument("--output-dir", type=str, help="Output directory (default: same directory as input)")
    
    # Input settings
    parser.add_argument("--is-gs", action="store_true", help="Whether input is Gaussian Splats (.ply format)")
    parser.add_argument("--opacity-threshold", type=float, default=0.01,
                       help="Only solid Gaussian Splats with opacities > threshold are used (default: 0.01)")
    parser.add_argument("--no-fingers", action="store_true", 
                       help="Whether the input model doesn't have ten separate fingers")
    parser.add_argument("--rest-pose", type=str, choices=["T-pose", "A-pose", "大-pose", "No"], default="No",
                       help="Specify the current rest pose of the input model")
    parser.add_argument("--rest-parts", type=str, nargs="*", 
                       choices=["Fingers", "Arms", "Legs", "Head"],
                       help="Specify which parts are already in T-pose")
    
    # Weight settings
    parser.add_argument("--use-normal", action="store_true",
                       help="Use normal information to improve performance (only works for meshes)")
    parser.add_argument("--no-bw-fix", action="store_false", dest="bw_fix",
                       help="Disable blend weight post-processing")
    parser.add_argument("--bw-vis-bone", type=str, default="LeftArm",
                       help="Bone name for weight visualization")
    
    # Animation settings
    parser.add_argument("--no-reset-to-rest", action="store_false", dest="reset_to_rest",
                       help="Don't apply predicted T-pose in final animatable model")
    parser.add_argument("--animation-file", type=str,
                       help="Path to animation file (.fbx) to apply to the model")
    parser.add_argument("--no-retarget", action="store_false", dest="retarget",
                       help="Disable animation retargeting")
    parser.add_argument("--no-inplace", action="store_false", dest="inplace",
                       help="Disable keeping looping animation in place")
    
    return parser.parse_args()

def extract_db(result):
    """Helper function to extract DB from a result dictionary"""
    return result[app.state]

def main():
    args = parse_args()
    
    print("💃 Make-It-Animatable - CLI")
    print("Initializing models...")
    
    # Initialize models and device
    init_models()
    
    db = DB()
    
    print(f"Processing input: {args.input}")
    
    # Step 1: Prepare input
    result = prepare_input(
        input_path=args.input,
        is_gs=args.is_gs,
        opacity_threshold=args.opacity_threshold,
        db=db,
        export_temp=False
    )
    db = extract_db(result)
    
    # If output directory was specified, update db's output_dir
    if args.output_dir:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        db.output_dir = output_dir
        
        # Update output paths
        db.joints_coarse_path = os.path.join(output_dir, "joints_coarse.glb")
        db.normed_path = os.path.join(output_dir, f"normed{os.path.splitext(args.input)[-1]}")
        db.sample_path = os.path.join(output_dir, "sample.glb")
        db.bw_path = os.path.join(output_dir, "bw.glb")
        db.joints_path = os.path.join(output_dir, "joints.glb")
        db.rest_lbs_path = os.path.join(output_dir, f"rest_lbs.{'ply' if args.is_gs else 'glb'}")
        db.rest_vis_path = os.path.join(output_dir, "rest.glb")
        input_filename = os.path.splitext(os.path.basename(args.input))[0]
        db.anim_path = os.path.join(output_dir, f"{input_filename}.{'blend' if args.is_gs else 'fbx'}")
        db.anim_vis_path = os.path.join(output_dir, f"{input_filename}.glb")
    
    print("Preprocessing...")
    # Step 2: Preprocess the model
    result = preprocess(db)
    db = extract_db(result)
    
    print("Running inference...")
    # Step 3: Run inference to get blend weights and joints
    result = infer(args.use_normal, db)
    db = extract_db(result)
    
    print("Visualizing results...")
    # Step 4: Visualize
    result = vis(args.bw_fix, args.bw_vis_bone, args.no_fingers, db)
    db = extract_db(result)
    
    print("Creating animation...")
    # Step 5: Generate Blender animation
    result = vis_blender(
        args.reset_to_rest, 
        args.no_fingers, 
        args.rest_pose,
        args.rest_parts, 
        args.animation_file,
        args.retarget, 
        args.inplace, 
        db
    )
    db = extract_db(result)
    
    print(f"Output animatable model: {db.anim_path}")
    print(f"All outputs stored in: {db.output_dir}")
    
    # Clean up
    clear(db)
    
    print("✅ Process completed successfully!")

if __name__ == "__main__":
    main()
