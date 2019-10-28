

import pandas as pd
import os
import tensorflow as tf
 
tf.logging.set_verbosity(tf.logging.INFO)
...
def build_model():
    ############
    ...
    return model
 
 
def save_model_for_production(model, version, path='prod_models'):
    tf.keras.backend.set_learning_phase(1)
    if not os.path.exists(path):
        os.mkdir(path)
    export_path = os.path.join(
        tf.compat.as_bytes(path),
        tf.compat.as_bytes(version))
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)
 
    model_input = tf.saved_model.utils.build_tensor_info(model.input)
    model_output = tf.saved_model.utils.build_tensor_info(model.output)
 
    prediction_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'inputs': model_input},
            outputs={'output': model_output},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
 
    with tf.keras.backend.get_session() as sess:
        builder.add_meta_graph_and_variables(
            sess=sess, tags=[tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                'predict':
                    prediction_signature,
            })
 
        builder.save()
 
 
if __name__ == '__main__':
    model_file = './my_model.h5'
    if (os.path.isfile(model_file)):
        print('model file detected. Loading.')
        model = tf.keras.models.load_model(model_file)
    else:
        print('No model file detected.  Starting from scratch.')
        model = build_model()
        model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
        model.save(model_file)
 
    model.fit(X_train, y_train, batch_size=100, epochs=1, validation_data=(X_test, y_test))
    model.summary()
 
    export_path = "tf-model"
    save_model_for_production(model, "1", export_path)
