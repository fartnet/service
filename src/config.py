WEIGHTS_DIR = '/home/ubuntu/fartnet/service/src/weights/'
WEIGHTS_NAME = 'model.ckpt-32188'
S3_BUCKET = 'fartnet'
S3_OBJ_META = WEIGHTS_NAME + '.meta'
S3_OBJ_INDEX = WEIGHTS_NAME + '.index'
S3_OBJ_DATA = WEIGHTS_NAME + '.data-00000-of-00001'
MODEL_WEIGHTS_META = WEIGHTS_DIR + 'infer/infer.meta'
MODEL_WEIGHTS_CKPT = WEIGHTS_DIR + WEIGHTS_NAME
FART_GEN_TIMER = 30  # seconds
FART_CLEANUP_TIMER = 0.05  # seconds
