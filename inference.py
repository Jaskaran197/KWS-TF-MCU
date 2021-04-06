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

import pyaudio
import wave
from array import array



FORMAT=pyaudio.paInt16
CHANNELS=1
RATE=16000
CHUNK=1024
CHUNK_THRESH=64
RECORD_SECONDS=1
FILE_NAME="RECORDING.wav"
while(True):
    audio=pyaudio.PyAudio() #instantiate the pyaudio

    #recording prerequisites
    stream=audio.open(format=FORMAT,channels=CHANNELS, 
                      rate=RATE,
                      input=True,
                      frames_per_buffer=CHUNK)

    #starting recording
    frames=[]
    while(True):
        old_data=stream.read(CHUNK_THRESH)
        data_chunk=array('h',old_data)
        vol=max(data_chunk)
        #print(old_data)
        if(vol>=600):
            print('Triggered')
            #frames.append(old_data)
            break
    for i in range(0,int(RATE/CHUNK*RECORD_SECONDS)):
        #frames.append(old_data)
        data=stream.read(CHUNK)
        data_chunk=array('h',data)
        vol=max(data_chunk)
        #if(vol>=300):
         #   print("something said")
        frames.append(data)
        #else:
            #print("nothing")
        #print("\n")


    #end of recording
    stream.stop_stream()
    stream.close()
    audio.terminate()
    #writing to file
    wavfile=wave.open(FILE_NAME,'wb')
    wavfile.setnchannels(CHANNELS)
    wavfile.setsampwidth(audio.get_sample_size(FORMAT))
    wavfile.setframerate(RATE)
    wavfile.writeframes(b''.join(frames))#append frames recorded to file
    wavfile.close()

    label_wav('RECORDING.wav','Pretrained_models\labels.txt','DNN_S.pb','wav_data:0','labels_softmax:0',1)
