from flask import Flask
from config import *
import tensorflow as tf
import numpy as np
import boto3

app = Flask(__name__)

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
    def __init__():
        # Load the graph
        tf.reset_default_graph()
        saver = tf.train.import_meta_graph(MODEL_WEIGHTS_META)
        self.graph = tf.get_default_graph()
        self.sess = tf.InteractiveSession()
        saver.restore(self.sess, MODEL_WEIGHTS_CKPT)

    def predict(self):
        # Create 50 random latent vectors z
        _z = (np.random.rand(50, 100) * 2.) - 1

        # Synthesize G(z)
        z = self.graph.get_tensor_by_name('z:0')
        G_z = self.graph.get_tensor_by_name('G_z:0')
        fart = self.sess.run(G_z, {z: _z})
        
        return fart

nn = fartNet()
@app.route("/")
def hello():
    #nn.predict()
    return "Hello, World!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
