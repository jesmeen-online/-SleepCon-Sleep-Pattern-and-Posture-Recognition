# -SleepCon-Sleep-Pattern-and-Posture-Recognition
Welcome to the SleepCon repository! This project focuses on the recognition of sleep patterns and postures, offering significant potential for various clinical applications. By monitoring sleep postures autonomously and continuously, SleepCon aims to provide valuable insights that can help reduce health risks.

# Overview
SleepCon is a deep learning model designed to train on threshold images obtained from any sensor, such as cameras, pressure sensors, or electrocardiograms. In this study, we utilized data from a camera installed above a mattress to capture body movements and postures while subjects were lying down. The system employs Convolutional Neural Networks (CNN) and several pre-processing steps to collect and analyze this data, accurately recognizing different sleep postures.

# Key Features
Data Acquisition: The camera captures the body posture of the subject on the mattress.
Deep Learning: The system uses CNNs to analyze the data and recognize sleep postures.
Posture Recognition: SleepCon can identify three major postures: left, right, and supine.
Real-time Application: A desktop application utilizes the stored SleepCon model to detect sleep postures in real-time.

# Performance
Classification Accuracy: Greater than 90%
Real-time Application Accuracy: 100% after experimentation with the SleepCon model
This repository includes the code and resources needed to implement and test the SleepCon model, as well as the desktop application for real-time posture detection. We hope this tool aids in your research or clinical applications related to sleep pattern and posture monitoring.

# Install
pip install opencv-python
