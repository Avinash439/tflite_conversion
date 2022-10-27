import tensorflow as tf
saved_model_dir = "tensorflow_S/Save_Model/"
output_model_dir = 'tensorflow_S/model_to22.tflite'
def normalmodel_to_tflitemodel(saved_model_dir,output_model_dir):
    # here we are loading our saved deep learning model
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir, signature_keys=['serving_default'])
    # In optimization we reduce the size and make same accuracy
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    # converting into the tflite model
    tflite_model = converter.convert()
    # saving in specific path 
    with open(output_model_dir, 'wb') as f:
        f.write(tflite_model)
    print("Done")