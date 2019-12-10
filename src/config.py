import os

WEIGHTS_NAME = 'model.ckpt-13405'
S3_BUCKET = 'fartnet'
S3_OBJ_META = WEIGHTS_NAME + '.meta'
S3_OBJ_INDEX = WEIGHTS_NAME + '.index'
S3_OBJ_DATA = WEIGHTS_NAME + '.data-00000-of-00001'
MODEL_WEIGHTS_META = os.path.expanduser('~/fartnet/service/src/weights/' + S3_OBJ_META)
MODEL_WEIGHTS_CKPT = os.path.expanduser('~/fartnet/service/src/weights/' + WEIGHTS_NAME)
