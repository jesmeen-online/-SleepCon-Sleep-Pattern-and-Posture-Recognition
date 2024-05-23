print('Loading dependencies')

import os
import numpy as np
import cv2
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D

url = 'http://192.168.0.100:8080/video'
IMAGEHEIGHT = 512
IMAGEWIDTH = 1920
ROIWIDTH = 256
ROIWIDTH = 256 + 256
LEFT = int(IMAGEWIDTH / 2 - ROIWIDTH / 2)
RIGHT = LEFT + ROIWIDTH - 256
TOP = int(IMAGEHEIGHT / 2 - ROIWIDTH / 2)
BOTTOM = TOP + ROIWIDTH
SCOREBOXWIDTH = 1024
BARCHARTLENGTH = SCOREBOXWIDTH - 50
BARCHARTTHICKNESS = 15
BARCHARTGAP = 20
BARCHARTOFFSET = 8
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Model variables
NUMBEROFPOSTURES = 3
WEIGHTS_URL = 'my_model_weights_256_50011.h5'
POSTURE_ENCODING = {0: 'left', 1: 'right', 2: 'supine'}

# OpenCV image processing variables
BGSUBTHRESHOLD = 50
THRESHOLD = 50

# POSTURE Mode variables
POSTUREMODE = False  # Don't ever edit this!
POSTURES_RECORDED = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]

# Data Collection Mode variables
DATAMODE = False  # Don't ever edit this!
WHERE = "all"
POSTURE = "left"
NUMBERTOCAPTURE = 100

# Testing Predictions of Model Mode variables
PREDICT = False  # Don't ever edit this!
HISTORIC_PREDICTIONS = [np.ones((1, 8)), np.ones((1, 8)), np.ones((1, 8)), np.ones((1, 8)), np.ones((1, 8))]
IMAGEAVERAGING = 5

'''
USEFUL FUNCTIONS
'''


# Creating a path for storing data
def create_path(WHERE, POSTURE):
    print("Creating path to store data for collection...")
    DIR_NAME = f"./data/{WHERE}/{POSTURE}"
    print(DIR_NAME)

    if not os.path.exists(DIR_NAME):
        os.makedirs(DIR_NAME)
    if len(os.listdir(DIR_NAME)) == 0:
        img_label = int(1)
    else:
        img_label = int(float(sorted(os.listdir(DIR_NAME), key=len)[-1][:-4]))
    return img_label


# Creating our deep learning model to recognize the posture image
def create_model_for_stored_weight(outputSize):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(256, 256, 1)))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Flatten())
    model.add(Dropout(rate=0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(units=outputSize, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# Function to load the model
def load_model(weight_url):
    # Loading the model
    modelName = 'Sleeping Posture Recognition'
    print(f'Loading model {modelName}')
    model = create_model_for_stored_weight(NUMBEROFPOSTURES)
    model.load_weights(weight_url)

    return model, modelName


# Function that counts the number of times the last element is repeated (starting at the end of the array)
# without any gap. Returns the count and the percentage (count/length of array)
# For example [1, 1, 1, 2] would return 1, 0.25 while [1,1,2,2] would return 2, 0.5
# [1,1,1,1] would return 4, 1
def find_last_rep(array):
    last_element = array[-1]
    count = 0
    for ele in reversed(array):
        if last_element == ele:
            count += 1
        else:
            return count, count / len(array)
    return count, count / len(array)


# Draw frame to side of video capture. Populate this frame with front
# end for POSTURE and prediction modes
def drawSideFrame(historic_predictions, frame, modelName, label):
    global POSTURES_RECORDED
    # Streaming
    dataText = "Streaming..."

    # Creating the score frame with white background
    score_frame = np.ones((IMAGEHEIGHT, SCOREBOXWIDTH - 612, 3), np.uint8) * 255

    # POSTURE MODE front end stuff
    if POSTUREMODE:
        POSTURES_RECORDED.append(label)
        POSTURES_RECORDED = POSTURES_RECORDED[-10:]
        count, percent_finished = find_last_rep(POSTURES_RECORDED)

        # If the array recording the timeline of POSTUREs only contains one POSTURE
        if len(set(POSTURES_RECORDED)) == 1:
            # See the command
            pass

        # Colors of the bar graph showing progress of the POSTURE recognition
        if percent_finished == 1:
            color = (0, 204, 102)
        else:
            color = (20, 20, 220)

        FONT = cv2.FONT_HERSHEY_SIMPLEX
        # Drawing the bar chart
        start_pixels = np.array([20, 175])
        # text = '{} ({}%)' .format(POSTURE_ENCODING[POSTURES_RECORDED[-1]], percent_finished*100)
        cv2.putText(score_frame, "", tuple(start_pixels), FONT, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        chart_start = start_pixels + np.array([0, BARCHARTOFFSET])
        length = int(percent_finished * BARCHARTLENGTH)
        chart_end = chart_start + np.array(
            [length, BARCHARTTHICKNESS])
        cv2.rectangle(score_frame, tuple(chart_start), tuple(chart_end), color, cv2.FILLED)

        # Define background color for text
        # Define positions and background rectangles for text
        texts = [
            ('Press G to go back', (20, 25)),
            (f'Model : {modelName}', (20, 50)),
            (f'Data source : {dataText}', (20, 75)),
            (f'Label : {POSTURE_ENCODING[label]}', (20, 125)),
            (f'Action : {POSTURE_ENCODING[label]}', (20, 150))
        ]

        FONT = cv2.FONT_HERSHEY_COMPLEX
        font_scale = 0.65
        font_thickness = 2
        # Draw background rectangles for better text visibility
        for text, position in texts:
            (text_width, text_height), baseline = cv2.getTextSize(text, FONT, font_scale, font_thickness)
            text_bottom_left = (position[0], position[1] + baseline)
            top_left = (position[0] - 5, position[1] - text_height - baseline - 5)
            bottom_right = (position[0] + text_width + 5, position[1] + baseline + 5)
            cv2.rectangle(score_frame, top_left, bottom_right, (255, 255, 255), -1)

        # Put the text with a better font type
        for text, position in texts:
            cv2.putText(score_frame, text, position, FONT, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)


    elif PREDICT:
        FONT = cv2.FONT_HERSHEY_COMPLEX
        font_scale = 0.55
        font_thickness = 2

        # Define positions and background rectangles for text
        texts = [
            ('Press P to stop testing predictions', (20, 25)),
            (f'Model : {modelName}', (20, 50)),
            (f'Data source : {dataText}', (20, 75))
        ]
        # Draw background rectangles for better text visibility
        for text, position in texts:
            (text_width, text_height), baseline = cv2.getTextSize(text, FONT, font_scale, font_thickness)
            text_bottom_left = (position[0], position[1] + baseline)
            top_left = (position[0] - 5, position[1] - text_height - baseline - 5)
            bottom_right = (position[0] + text_width + 5, position[1] + baseline + 5)
            cv2.rectangle(score_frame, top_left, bottom_right, (255, 255, 255), -1)

        # Put the text with a better font type
        for text, position in texts:
            cv2.putText(score_frame, text, position, FONT, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

        # Converting predictions into an array
        predictions = np.array(historic_predictions)

        # Taking a mean of historic predictions
        average_predictions = predictions.mean(axis=0)[0]
        sorted_args = list(np.argsort(average_predictions))

        # Drawing the prediction probabilities in a bar chart
        start_pixels = np.array([20, 150])
        for count, arg in enumerate(list(reversed(sorted_args))):

            probability = round(average_predictions[arg])

            predictedLabel = POSTURE_ENCODING[arg]

            if arg == label:
                color = (0, 204, 102)
            else:
                color = (20, 20, 220)

            text = '{}. {} ({}%)'.format(count + 1, predictedLabel, probability * 100)
            cv2.putText(score_frame, text, tuple(start_pixels), FONT, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
            chart_start = start_pixels + np.array([0, BARCHARTOFFSET])
            length = int(probability * BARCHARTLENGTH)
            chart_end = chart_start + np.array(
                [length, BARCHARTTHICKNESS])
            cv2.rectangle(score_frame, tuple(chart_start), tuple(chart_end), color, cv2.FILLED)

            start_pixels = start_pixels + np.array([0, BARCHARTGAP + BARCHARTTHICKNESS + BARCHARTOFFSET])

    # No mode active front end stuff
    else:

        FONT = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.65
        font_thickness = 2

        # Define positions and background rectangles for text
        texts = [
            ('Press P to test model', (20, 25)),
            ('Press G for POSTURE Mode', (20, 50)),
            ('Press R to reset background', (20, 75)),
            (f'Model : {modelName}', (20, 100)),
            (f'Data source : {dataText}', (20, 125))
        ]

        # Draw background rectangles for better text visibility
        for text, position in texts:
            (text_width, text_height), baseline = cv2.getTextSize(text, FONT, font_scale, font_thickness)
            text_bottom_left = (position[0], position[1] + baseline)
            top_left = (position[0] - 5, position[1] - text_height - baseline - 5)
            bottom_right = (position[0] + text_width + 5, position[1] + baseline + 5)
            cv2.rectangle(score_frame, top_left, bottom_right, (255, 255, 255), -1)

        # Put the text with a better font type
        for text, position in texts:
            cv2.putText(score_frame, text, position, FONT, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

    print(score_frame.shape)
    print(frame.shape)
    return np.hstack((score_frame, frame))


# Remove the background from a new frame
def remove_background(bgModel, frame):
    fgmask = bgModel.apply(frame, learningRate=0)
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=eroded)
    return res


# Show the processed, thresholded image of Sleep in side frame on right
def drawMask(frame, mask):
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # mask_frame = 200*np.ones((IMAGEHEIGHT,ROIWIDTH+20,3),np.uint8)
    mask_frame = np.ones((IMAGEHEIGHT, 350, 3), np.uint8)
    mask_frame[:512, 10:256 + 10] = mask
    cv2.putText(mask_frame, "Mask",
                (100, 290), FONT, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    print("check")
    print(frame.shape)
    print(mask_frame.shape)

    return np.hstack((frame, mask_frame))


# The controller/frontend that subtracts the background

# Create a VideoCapture object
def capture_background():
    # cap = cv2.VideoCapture(CAMERA)
    cap = cv2.VideoCapture(url)
    while True:
        ret, frame = cap.read()
        #
        if not ret:
            break
        FONT = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (1024, 512))
        # Text labels
        # Text labels with background
        cv2.rectangle(frame, (10, 10), (330, 150), (0, 0, 0), -1)  # Background rectangle for text
        cv2.putText(frame, "Press B to capture background", (20, 50), FONT, font_scale, (255, 255, 255), font_thickness,
                    cv2.LINE_AA)
        cv2.putText(frame, "(Set the sleeping area within the red frame)", (20, 90), FONT, font_scale * 0.7,
                    (255, 255, 255), font_thickness, cv2.LINE_AA)
        cv2.putText(frame, "Press Q to quit.", (20, 130), FONT, font_scale, (255, 255, 255), font_thickness,
                    cv2.LINE_AA)

        # Rectangle around sleeping area
        cv2.rectangle(frame, (LEFT, TOP), (RIGHT, BOTTOM), (0, 0, 255), 4)  # Red color, thicker border
        # Add text "Sleeping Area" above the rectangle
        text_size = cv2.getTextSize("Sleeping Area", FONT, font_scale * 0.7, font_thickness)[0]
        text_x = LEFT + (RIGHT - LEFT - text_size[0]) // 2  # Center text horizontally
        cv2.putText(frame, "Sleeping Area -->", (520, 330), FONT, font_scale * 0.7, (0, 0, 255), font_thickness,
                    cv2.LINE_AA)

        cv2.imshow('Welcome: Capture Background', frame)

        k = cv2.waitKey(5)

        # If key b is pressed
        if k == ord('b'):
            bgModel = cv2.createBackgroundSubtractorMOG2(0, BGSUBTHRESHOLD)
            # cap.release()
            cv2.destroyAllWindows()
            # break
            return bgModel

        # If key q is pressed
        elif k == ord('q'):
            bgModel = None
            cap.release()
            cv2.destroyAllWindows()
            # break
            return bgModel


if __name__ == '__main__':
    # Create a path for the data collection
    img_label = create_path(WHERE, POSTURE)

    # Load dependencies
    model, modelName = load_model(WEIGHTS_URL)

    # Background capture model
    bgModel = capture_background()

    # If a background has been captured
    if bgModel:

        cap = cv2.VideoCapture(url)

        while True:
            # Capture frame
            label, frame = cap.read()

            # Flip frame
            frame = cv2.flip(frame, 1)

            frame = cv2.resize(frame, (1024, 512))
            # Applying smoothing filter that keeps edges sharp
            frame = cv2.bilateralFilter(frame, 5, 50, 100)

            cv2.rectangle(frame, (LEFT, TOP), (RIGHT - 4, BOTTOM - 4), (255, 0, 0), 3)

            # Remove background
            no_background = remove_background(bgModel, frame)

            # Selecting region of interest
            roi = no_background[TOP:BOTTOM, LEFT:RIGHT]

            # Converting image to gray
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # Blurring the image
            blur = cv2.GaussianBlur(gray, (41, 41), 0)

            # Thresholding the image
            ret, thresh = cv2.threshold(blur, THRESHOLD, 255, cv2.THRESH_BINARY)
            img_RGB_smaller = cv2.resize(src=thresh, dsize=(256, 256))
            img_RGB_smaller.shape
            # Predicting and storing predictions
            print(model.summary())
            prediction = model.predict(img_RGB_smaller.reshape(1, 256, 256, 1) / (255))
            prediction_final = np.argmax(prediction)
            HISTORIC_PREDICTIONS.append(prediction)
            HISTORIC_PREDICTIONS = HISTORIC_PREDICTIONS[-IMAGEAVERAGING:]

            # Draw new frame with graphs
            new_frame = drawSideFrame(HISTORIC_PREDICTIONS, frame, 'POSTURE Model', prediction_final)

            # Draw new dataframe with mask
            new_frame = drawMask(new_frame, thresh)

            # If Datamode
            if DATAMODE:
                time.sleep(0.03)
                cv2.imwrite(f"./data/{WHERE}/{POSTURE}" + f"/{img_label}.png", thresh)
                cv2.putText(new_frame, "Photos Captured:", (980, 400), FONT, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(new_frame, f"{i}/{NUMBERTOCAPTURE}", (1010, 430), FONT, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
                img_label += 1
                i += 1
                if i > NUMBERTOCAPTURE:
                    cv2.putText(new_frame, "Done!", (980, 400), FONT, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
                    DATAMODE = False
                    i = None
            else:
                cv2.putText(new_frame, "Sleeping Area->>", (980, 375), FONT, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                # cv2.putText(new_frame, "Press D to collect", (980,375), FONT, 0.6, (0,0,0), 2, cv2.LINE_AA)
                # cv2.putText(new_frame, f"{NUMBERTOCAPTURE} {WHERE} images", (980,400), FONT, 0.6, (0,0,0), 2, cv2.LINE_AA)
                # cv2.putText(new_frame, f"for POSTURE {POSTURE}", (980,425), FONT, 0.6, (0,0,0), 2, cv2.LINE_AA)

            # Show the frame
            cv2.imshow('SLEEPING POSTURE', new_frame)

            key = cv2.waitKey(5)

            # If q is pressed, quit the app+

            if key == ord('q'):
                break

            # If r is pressed, reset the background
            if key == ord('r'):
                PREDICT = False
                DATAMODE = False
                cap.release()
                cv2.destroyAllWindows()
                bgModel = capture_background()
                cap = cv2.VideoCapture(url)

            # If d is pressed, go into to data collection mode
            if key == ord('d'):
                PREDICT = False
                POSTUREMODE = False
                DATAMODE = True
                i = 1

            # If p is pressed, predict
            if key == ord('p'):
                POSTUREMODE = False
                DATAMODE = False
                PREDICT = not PREDICT

            # If g is pressed go into POSTURE mode
            if key == ord('g'):
                DATAMODE = False
                PREDICT = False
                POSTUREMODE = not POSTUREMODE

        # Release the cap and close all windows if loop is broken
        cap.release()
        cv2.destroyAllWindows()
