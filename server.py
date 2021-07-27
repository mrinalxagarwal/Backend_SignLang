import base64
import io

import cv2
import numpy as np
from PIL import Image
from flask import Flask, render_template
from flask_socketio import SocketIO, emit, join_room, leave_room, send
from tensorflow.keras.models import load_model

# global variables
labels = {
    0: 'book',
    1: 'chair',
    2: 'clothes',
    3: 'computer',
    4: 'drink',
    5: 'drum',
    6: 'family',
    7: 'football',
    8: 'go',
    9: 'hat',
    10: 'hello',
    11: 'kiss',
    12: 'like',
    13: 'play',
    14: 'school',
    15: 'street',
    16: 'table',
    17: 'university',
    18: 'violin',
    19: 'wall'
}

app = Flask(__name__)
socketio = SocketIO(app)


@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template('index.html')


@socketio.on('image')
def image(data_image):
    frame_buffer = np.empty((0, *dim, channels))

    # unpack request
    for data in data_image:
        # decode and convert the request in image
        img = Image.open(io.BytesIO(base64.b64decode(data[23:])))
        # converting RGB to BGR 
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        # process the frame
        frame_res = cv2.resize(frame, dim)
        frame_res = frame_res / 255.0
        # append the frame to buffer
        frame_resh = np.reshape(frame_res, (1, *frame_res.shape))
        frame_buffer = np.append(frame_buffer, frame_resh, axis=0)

    frame_buffer_resh = frame_buffer.reshape(1, *frame_buffer.shape)
    # model prediction
    predictions = model.predict(frame_buffer_resh)[0]
    #bestpred
    best_pred_idx = np.argmax(predictions)
    acc_best_pred = predictions[best_pred_idx]
    #mislabel check
    if acc_best_pred > threshold:
        gloss = labels[best_pred_idx]
    else:
        gloss = 'none'
    # send the predicted gloss to the client
    send(gloss.upper(), to= data['room'])

@socketio.on('join')
def on_join(data):
    username = data['username']
    room = data['room']
    join_room(room)
    send(username + ' has entered the room.', to=room)

@socketio.on('leave')
def on_leave(data):
    username = data['username']
    room = data['room']
    leave_room(room)
    send(username + ' has left the room.', to=room)

if __name__ == '__main__':
    #configs
    dim = (224, 224)
    frames = 10
    channels = 3
    model_path = './model/WLASL20c_model.h5'
    threshold = .50

    #loading model
    model = load_model(model_path)

    #start app
    socketio.run(app=app, host='127.0.0.1', port=5000)
