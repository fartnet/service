from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask import Flask, send_from_directory
import os
from config import *
import tensorflow as tf
import librosa
import numpy as np
import boto3

tf.logging.set_verbosity(tf.logging.ERROR)

app = Flask(__name__)

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["1 per minute"]
)

# Get the weights from s3 so we can update them on the
# fly and just reboot the server
s3 = boto3.resource('s3')

if not os.path.exists(WEIGHTS_DIR + S3_OBJ_META):
    s3.Bucket(S3_BUCKET).download_file(S3_OBJ_META, WEIGHTS_DIR + S3_OBJ_META)
if not os.path.exists(WEIGHTS_DIR + S3_OBJ_INDEX):
    s3.Bucket(S3_BUCKET).download_file(S3_OBJ_INDEX, WEIGHTS_DIR + S3_OBJ_INDEX)
if not os.path.exists(WEIGHTS_DIR + S3_OBJ_DATA):
    s3.Bucket(S3_BUCKET).download_file(S3_OBJ_DATA, WEIGHTS_DIR + S3_OBJ_DATA)

class fartNet:
    def __init__(self):
        # Load the model
        tf.reset_default_graph()
        self.saver = tf.train.import_meta_graph(MODEL_WEIGHTS_META)
        self.graph = tf.get_default_graph()
        self.sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True,
                                                                log_device_placement=True))
        self.saver.restore(self.sess, MODEL_WEIGHTS_CKPT)
        print('********DEBUG:********')
        print(MODEL_WEIGHTS_META)
        print(MODEL_WEIGHTS_CKPT)

    def predict(self):
        # Create 50 random latent vectors z
        _z = (np.random.rand(50, 100) * 2.) - 1

        # Synthesize G(z)
        for name in [n.name for n in tf.get_default_graph().as_graph_def().node]:
            print(name)
        z = self.graph.get_tensor_by_name('z:0')
        G_z = self.graph.get_tensor_by_name('G_z:0')
        fart = self.sess.run(G_z, {z: _z})

        return fart

nn = fartNet()
fartdir = '/home/ubuntu/fartnet/generated_farts/'
global fart_counter
fart_counter = 0  # should think of a better way to do this that is robust to shutdowns
@app.route("/")
def hello():
    return "Hello, World!"

@app.route("/getfart" ,methods=['GET'])
@limiter.limit("1/second", error_message='chill, making you 50 farts already!')
def get_fart():
    global fart_counter
    fart_numpy = nn.predict()
    filename = 'fart_%d'%(fart_counter) + '.wav'
    librosa.output.write_wav(fartdir + filename, fart_numpy.ravel(), 16000)
    fart_counter += 1
    return send_from_directory(fartdir, filename, as_attachment=True, cache_timeout=0)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
