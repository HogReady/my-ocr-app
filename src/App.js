import React, { useState, useEffect, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';

const OCRApp = () => {
  const [detectorModel, setDetectorModel] = useState(null);
  const [recognizerModel, setRecognizerModel] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const canvasRef = useRef(null);

  useEffect(() => {
    const loadModels = async () => {
      try {
        const detector = await tf.loadGraphModel('web_model/detector_model/model.json');
        const recognizer = await tf.loadGraphModel('web_model/recognizer_model/model.json');
        setDetectorModel(detector);
        setRecognizerModel(recognizer);
      } catch (error) {
        console.error('Error loading models:', error);
      }
    };

    loadModels();
  }, []);

  const handleImageUpload = async (event) => {
    const file = event.target.files[0];
    if (file && file.type.startsWith('image/')) {
      const image = await loadImage(file);
      const canvas = canvasRef.current;
      canvas.width = image.width;
      canvas.height = image.height;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
      await detectText(canvas);
    } else {
      console.error('Uploaded file is not an image.');
    }
  };

  const loadImage = (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const img = new Image();
        img.src = reader.result;
        img.onload = () => resolve(img);
      };
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  };

  const detectText = async (canvas) => {
    if (!detectorModel || !recognizerModel) return;

    // Convert canvas to tensor
    const input = tf.browser.fromPixels(canvas).expandDims(0).toFloat();
    // Normalize input
    const normalized = input.div(255.0);

    // Run the detection model
    const detectionResult = await detectorModel.executeAsync(normalized);

    // Process the detection result to get bounding boxes
    const boxes = await processDetectionResult(detectionResult);

    // Recognize text in each detected box
    const texts = await recognizeText(boxes, canvas);

    setPredictions(texts);

    // Dispose tensors to release memory
    tf.dispose([input, normalized, detectionResult]);
  };

  const processDetectionResult = async (detectionResult) => {
    // This is an example of how to process detection results
    // Extract bounding boxes from detectionResult
    // Replace this logic with actual processing based on your model's output
    const [boxes] = detectionResult;
    const boxArr = boxes.arraySync();
    return boxArr.map(box => ({
      x: box[1] * canvasRef.current.width,
      y: box[0] * canvasRef.current.height,
      width: (box[3] - box[1]) * canvasRef.current.width,
      height: (box[2] - box[0]) * canvasRef.current.height,
    }));
  };

  const recognizeText = async (boxes, canvas) => {
    const ctx = canvas.getContext('2d');
    const texts = [];

    for (const box of boxes) {
      const { x, y, width, height } = box;
      const imageData = ctx.getImageData(x, y, width, height);
      const input = tf.browser.fromPixels(imageData).expandDims(0).toFloat();
      const normalized = input.div(255.0);

      // Run the recognizer model
      const recognizerResult = await recognizerModel.executeAsync(normalized);

      // Process the recognizer result to get the text
      // This should be replaced with actual processing logic based on your model's output
      const text = await processRecognizerResult(recognizerResult);

      texts.push({ text });

      // Dispose tensors to release memory
      tf.dispose([input, normalized, recognizerResult]);
    }

    return texts;
  };

  const processRecognizerResult = async (recognizerResult) => {
    // Example of how to process recognizer results to get the text
    // Replace this with actual processing logic based on your model's output
    const [textTensor] = recognizerResult;
    const text = textTensor.dataSync().join('');
    return text;
  };

  return (
    <div>
      <h1>OCR App</h1>
      <input type="file" accept="image/*" onChange={handleImageUpload} />
      <canvas ref={canvasRef} style={{ display: 'none' }}></canvas>
      {predictions.map((prediction, index) => (
        <div key={index}>
          <p>{prediction.text}</p>
        </div>
      ))}
    </div>
  );
};

export default OCRApp;
