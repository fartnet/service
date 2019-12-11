import threading
import atexit
import os
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask import Flask, send_from_directory
from config import *
import tensorflow as tf
import librosa
import numpy as np
import boto3

tf.logging.set_verbosity(tf.logging.ERROR)

dataLock = threading.Lock()
checker_thread = threading.Thread()

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
delete_list = []

def create_app():
    app = Flask(__name__)

    limiter = Limiter(
        app,
        key_func=get_remote_address,
        default_limits=["1 per minute"]
    )

    @app.route("/")
    def hello():
        return "Hello, World!"

    @app.route("/getfart" ,methods=['GET'])
    @limiter.limit("1/second", error_message='chill, making you 50 farts already!')
    def get_fart():
        global delete_list
        if len(os.listdir(fartdir)) == 0:
            generate_farts()
        fart_to_fetch = int(sorted(os.listdir(fartdir))[-1][5:-4])  # fart with lowest # in name
        filename = 'fart_%d'%(fart_to_fetch) + '.wav'
        delete_list.append(fartdir + filename)
        return send_from_directory(fartdir, filename, as_attachment=True, cache_timeout=0)

    def delete_spent_farts_and_generate_new_farts(): #fartfactory
        global delete_list
        if len(delete_list > 0):
            for wavfile in delete_list:
                os.remove(wavfile)
        delete_list = []
        if len(os.listdir(fartdir)) == 0:
            last_fart = 0
        elif len(os.listdir(fartdir)) < 200:
            last_fart = int(sorted(os.listdir(fartdir))[0][5:-4])
        for i in range(4):
            farts_numpy = nn.predict()
            for j in range(50):
                last_fart += 1
                filename = 'fart_%d'%(last_fart) + '.wav'
                librosa.output.write_wav(fartdir + filename, farts_numpy[j,:,:].ravel(), 16000)

    def interrupt():
        global checker_thread
        checker_thread.cancel()

    def start_checker_thread():
        global checker_thread
        checker_thread = threading.Timer(FART_GEN_CHECKER_TIMER, delete_spent_farts_and_generate_new_farts, ())
        checker_thread.start()

    start_checker_thread()
    atexit.register(interrupt)
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=80)
