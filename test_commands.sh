curl -X POST http://localhost:8080/rig   -H 'Content-Type: application/json'   -d '{
        "input_uri": "gs://mia-uploads/cbr.glb",
        "animation_uri": "",
        "config": {
          "is_gs": false,
          "rest_pose": "No",
          "rest_parts": [],
          "retarget": false
        }
      }'

curl -X POST http://localhost:8080/rig \
  -H "Content-Type: application/json" \
  -d '{
        "input_uri": "gs://mia-uploads/cbr_meshy.glb",
        "animation_uri": "gs://mia-uploads/run.fbx",
        "config": {
          "is_gs": false,
          "rest_pose": "No",
          "rest_parts": []
        }
      }'
