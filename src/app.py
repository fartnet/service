from flask import Flask
import tensorflow as tf
import os

app = Flask(__name__)

MODEL_WEIGHTS_META = os.path.expanduser('~/fartnet_weights/infer.meta')
MODEL_WEIGHTS_CKPT = os.path.expanduser('~/fartnet_weights/model.ckpt')

# Load the graph
tf.reset_default_graph()
saver = tf.train.import_meta_graph(MODEL_WEIGHTS_META)
graph = tf.get_default_graph()
sess = tf.InteractiveSession()
saver.restore(sess, MODEL_WEIGHTS_CKPT)

def predict():
    # Create 50 random latent vectors z
    _z = (np.random.rand(50, 100) * 2.) - 1

    # Synthesize G(z)
    z = graph.get_tensor_by_name('z:0')
    G_z = graph.get_tensor_by_name('G_z:0')
    fart = sess.run(G_z, {z: _z})
    
    return fart

@app.route("/")
def hello():
    return "Hello, World!"
