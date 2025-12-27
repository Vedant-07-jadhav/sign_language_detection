# Sign_Language_Detection

## Preprocessing
 ### Preprocessing pipeline is as follows:
   video → clip frames → bbox crop → resize → normalize → sample frames

   - Crop using BBox
   - Resize to 112×112
   - frame sampling: 16 frames 

- Models to be used:
   - yolov12
      - Yolo can't be used directly as it need to implement with RNN or LSTM to have knowledge of previous frames for each data video.
   - CNN 
   - 2D CNN
   - 3D CNN
      - C3D
      - I3D
      - R(2+1)D
   - Hybrid CNN-RNN/ LSTM
   - Two-stream CNN
   - Two-Stream Inflated 3D ConvNet (I3D)
   - Vision transformer(ViT)
   - Action Transformer (e.g., VideoBERT)


- First learn about all the things related to to how to implement the all the Neural networks and also how to make a pipeline
- complete the CS231n lectures to get clear ideas
