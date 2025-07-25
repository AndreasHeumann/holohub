# UCX-based Distributed Endoscopy Tool Tracking

This application is similar to the Endoscopy Tool Tracking application, but the distributed version divides the application into three fragments:

1. Video Input: get video input from a pre-recorded video file.
2. Inference: run the inference using LSTM and run the post-processing script.
3. Visualization: display input video and inference results.

Based on an LSTM (long-short term memory) stateful model, these applications demonstrate the use of custom components for tool tracking, including composition and rendering of text, tool position, and mask (as heatmap) combined with the original video stream.

## Requirements

The provided applications are configured to use a pre-recorded endoscopy video (replayer).

## Data

[📦️ (NGC) Sample App Data for AI-based Endoscopy Tool Tracking](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/holoscan_endoscopy_sample_data)

The data is automatically downloaded and converted to the correct format when building the application.
If you want to manually convert the video data, please refer to the instructions for using the [convert_video_to_gxf_entities](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/scripts#convert_video_to_gxf_entitiespy) script.

## Run Instructions


```sh
# Build the Holohub container for the Distributed Endoscopy Tool Tracking application
./holohub build-container ucx_endoscopy_tool_tracking --img holohub:ucx_endoscopy_tool_tracking

# Launch the container
./holohub run-container ucx_endoscopy_tool_tracking --no-docker-build --img holohub:ucx_endoscopy_tool_tracking

# Build the Distributed Endoscopy Tool Tracking application
./holohub build ucx_endoscopy_tool_tracking --local

# Generate the TRT engine file from onnx
python3 utilities/generate_trt_engine.py --input data/endoscopy/tool_loc_convlstm.onnx --output data/endoscopy/engines/ --fp16

# Start the application with all three fragments
./holohub run ucx_endoscopy_tool_tracking --language=cpp --local --no-local-build

# Once you have completed the step to generate the TRT engine file, you may exit the container and
#  use the following commands to run the application in distributed mode:

# Start the application with the video_in fragment
./holohub run ucx_endoscopy_tool_tracking --language=cpp --run-args="--driver --worker --fragments video_in --address :9999"
# Start the application with the inference fragment
./holohub run ucx_endoscopy_tool_tracking --language=cpp --run-args="--worker --fragments inference --address :9999"
# Start the application with the visualization fragment
./holohub run ucx_endoscopy_tool_tracking --language=cpp --run-args="--worker --fragments viz --address :9999"
```
