# Sign_Language_Detection

- Preprocess all the datasets and convert into csv
- Crop videos to there bboxes and convert in grayscale as we need only motion and movents of the hands for sign language detection
- Models to be used:
   - yolov12
      - Yolo can't be used directly as it need to implement with RNN or LSTM to have knowledge of previous frames for each data video.
   - CNN 
   - 2D CNN
   - 3D CNN
   - Hybrid CNN-RNN/ LSTM
   - Two-stream CNN
   - Vision transformer

- First learn about all the things related to to how to implement the all the Neural networks and also how to make a pipeline
- complete the CS231n lectures to get clear ideas
