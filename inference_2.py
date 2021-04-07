from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf

# pylint: disable=unused-import
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
# pylint: enable=unused-import

FLAGS = None


def load_graph(filename):
  """Unpersists graph from file as default graph."""
  with tf.gfile.FastGFile(filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')


def load_labels(filename):
  """Read in labels, one label per line."""
  return [line.rstrip() for line in tf.gfile.GFile(filename)]


def run_graph(wav_data, labels, input_layer_name, output_layer_name,
              num_top_predictions):
  """Runs the audio data through the graph and prints predictions."""
  with tf.Session() as sess:
    # Feed the audio data as input to the graph.
    #   predictions  will contain a two-dimensional array, where one
    #   dimension represents the input image count, and the other has
    #   predictions per class
    softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)
    #print(wav_data)
    predictions, = sess.run(softmax_tensor, {input_layer_name: wav_data})
    #print(len(predictions))
    # Sort to show labels in order of confidence
    top_k = predictions.argsort()[-num_top_predictions:][::-1]
    for node_id in top_k:
      human_string = labels[node_id]
      score = predictions[node_id]
      
      print('%s (score = %.5f)' % (human_string, score))

    return 0


def label_wav(wav, labels, graph, input_name, output_name, how_many_labels):
  """Loads the model and labels, and runs the inference to print predictions."""
  if not wav or not tf.gfile.Exists(wav):
    tf.logging.fatal('Audio file does not exist %s', wav)

  if not labels or not tf.gfile.Exists(labels):
    tf.logging.fatal('Labels file does not exist %s', labels)

  if not graph or not tf.gfile.Exists(graph):
    tf.logging.fatal('Graph file does not exist %s', graph)

  labels_list = load_labels(labels)

  # load graph, which is stored in the default session
  load_graph(graph)

  with open(wav, 'rb') as wav_file:
    wav_data = wav_file.read()
  #print(wav_data)
  #print(len(wav_data))
  #print(type(wav_data))

  #print(labels_list)
  #print(input_name, output_name, how_many_labels)
  run_graph(wav_data, labels_list, input_name, output_name, how_many_labels)

SILENCE_LIMIT = 1  # Silence limit in seconds. The max ammount of seconds where
                   # only silence is recorded. When this time passes the
                   # recording finishes and the file is delivered.

PREV_AUDIO = 0.2  # Previous audio (in seconds) to prepend. When noise
                  # is detected, how much of previously recorded audio is
                  # prepended. This helps to prevent chopping the beggining
                  # of the phrase.
import math
import audioop
import time
import pyaudio
from collections import deque
import wave
import os
FORMAT=pyaudio.paInt16
CHANNELS=1
RATE=16000
CHUNK=1024
RECORD_SECONDS=2
def listen_for_speech(threshold=400, num_phrases=30):
    THRESHOLD=400
    """
    Listens to Microphone, extracts phrases from it and sends it to 
    model and returns response. a "phrase" is sound 
    surrounded by silence (according to threshold). num_phrases controls
    how many phrases to process before finishing the listening process 
    (-1 for infinite). 
    """

    #Open stream
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print ("* Listening mic. ")
    audio2send = []
    cur_data = ''  # current chunk  of audio data
    rel = RATE/CHUNK
    slid_win = deque(maxlen=int(SILENCE_LIMIT * rel))
    #Prepend audio from 0.5 seconds before noise was detected
    prev_audio = deque(maxlen=int(PREV_AUDIO * rel)) 
    started = False
    n = num_phrases
    response = []

    while (num_phrases == -1 or n > 0):
        cur_data = stream.read(CHUNK)
        slid_win.append(math.sqrt(abs(audioop.avg(cur_data, 4))))
        #print slid_win[-1]
        if(sum([x > THRESHOLD for x in slid_win]) > 0):
            if(not started):
                print ("Triggered")
                started = True
            audio2send.append(cur_data)
        elif (started is True):
            print ("You said.. ")
            # The limit was reached, finish capture and deliver.
            filename = save_speech(list(prev_audio) + audio2send, p)
            label_wav(filename,'Pretrained_models\labels4.txt','dnn_48_up_down_left_right.pb','wav_data:0','labels_softmax:0',1)

            # Send file to Google and get response
            #r = stt_google_wav(filename) 
            #if num_phrases == -1:
                #print ("Response", r)
            #else:
                #response.append(r)
            # Remove temp file. Comment line to review.
            os.remove(filename)
            # Reset all
            started = False
            slid_win = deque(maxlen=int(SILENCE_LIMIT * rel))
            prev_audio = deque(maxlen=int(0.5 * rel)) 
            audio2send = []
            n -= 1
            print ("Listening ...")
        else:
            prev_audio.append(cur_data)

    print ("* Done recording")
    stream.close()
    p.terminate()

    return response


def save_speech(data, p):
    """ Saves mic data to temporary WAV file. Returns filename of saved 
        file """

    filename = 'output_'+str(int(time.time()))
    # writes data to WAV file
    #data = ''.join(data)
    
    wf = wave.open(filename + '.wav', 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)  # TODO make this value a function parameter?
    #wf.writeframes(data)
    wf.writeframes(b''.join(data))
    wf.close()
    return filename + '.wav'

listen_for_speech()
