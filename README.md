# Dual-Hand Sign Language Detector Pipeline

An end-to-end Machine Learning pipeline running locally on Apple Silicon that captures dual-hand landmarks, processes architectural coordinate matrices, and uses a Convolutional Neural Network (CNN) to predict custom gestures in real-time.

## Project Architecture
* **Data Collection Engine:** Captures and normalizes structural geometric frames using the modern MediaPipe Tasks API.
* **Neural Network:** A custom Keras CNN architecture optimized for spatial relationship learning on a coordinate vector.
* **Live Inference Engine:** Real-time classification overlay on a live webcam feed.

## Scalability Note
This project is currently configured as a functional **Minimum Viable Product (MVP)** demonstrating complete pipeline execution using two custom sign templates. The architecture is modular and designed to instantly scale to the full 26-letter ASL alphabet simply by expanding the gesture dataset matrix array.
