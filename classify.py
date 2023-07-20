# from tensorflow.python.platform import gfile
import sys
import os
import pathlib


# speicherorte fuer trainierten graph und labels in train.sh festlegen ##

# Disable tensorflow compilation warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow.python.util import deprecation


if __name__ == "__main__":
    # Supress deprecation warnings
    deprecation._PRINT_DEPRECATION_WARNINGS = False
    os.environ['AUTOGRAPH_VERBOSITY'] = "0"

    base_path = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_path, "tf_files")

    image_path = sys.argv[1]
    path = pathlib.Path(image_path)
    # angabe in console als argument nach dem aufruf

    if path.is_dir():

        # holt labels aus file in array
        label_lines = [
            line.rstrip()
            for line in tf.compat.v1.gfile.GFile(
                os.path.join(data_dir, "retrained_labels.txt")
            )
        ]
        # !! labels befinden sich jeweils in eigenen lines -> keine aenderung in retrain.py noetig -> falsche darstellung im windows editor !!

        # graph einlesen, wurde in train.sh -> call retrain.py trainiert
        with tf.compat.v1.gfile.FastGFile(
            os.path.join(data_dir, "retrained_graph.pb"), "rb"
        ) as f:
            graph_def = (
                tf.compat.v1.GraphDef()
            )  # The graph-graph_def is a saved copy of a TensorFlow graph; objektinitialisierung
            graph_def.ParseFromString(
                f.read()
            )  # Parse serialized protocol buffer data into variable
            _ = tf.import_graph_def(
                graph_def, name=""
            )  # import a serialized TensorFlow GraphDef protocol buffer, extract objects in the GraphDef as tf.Tensor

            # https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/inception.py ; ab zeile 276

            with tf.compat.v1.Session() as sess:
                softmax_tensor = sess.graph.get_tensor_by_name("final_result:0")
                # return: Tensor("final_result:0", shape=(?, 4), dtype=float32); stringname definiert in retrain.py, zeile 1064
                for img in os.listdir(path):
                    image_data = tf.compat.v1.gfile.FastGFile(f"{path}/{img}", "rb").read()
                    predictions = sess.run(
                        softmax_tensor, {"DecodeJpeg/contents:0": image_data}
                    )
                    # gibt prediction values in array zuerueck:

                    top_k = predictions[0].argsort()[-len(predictions[0]) :][::-1]
                    # sortierung; circle -> 0, plus -> 1, square -> 2, triangle -> 3; array return bsp [3 1 2 0] -> sortiert nach groesster uebereinstimmmung

                    # output
                    result = {"Image": f"{path}/{img}"}
                    for node_id in top_k:
                        human_string = label_lines[node_id]
                        score = predictions[0][node_id]
                        result[human_string] = score

                    # score = max(result[human_string], key=lambda x:result[human_string])
                    print(result)

    if path.is_file():
        # bilddatei readen
        image_data = tf.compat.v1.gfile.FastGFile(image_path, "rb").read()

        # holt labels aus file in array
        label_lines = [
            line.rstrip()
            for line in tf.compat.v1.gfile.GFile(
                os.path.join(data_dir, "retrained_labels.txt")
            )
        ]
        # !! labels befinden sich jeweils in eigenen lines -> keine aenderung in retrain.py noetig -> falsche darstellung im windows editor !!

        # graph einlesen, wurde in train.sh -> call retrain.py trainiert
        with tf.compat.v1.gfile.FastGFile(
            os.path.join(data_dir, "retrained_graph.pb"), "rb"
        ) as f:
            graph_def = (
                tf.compat.v1.GraphDef()
            )  ## The graph-graph_def is a saved copy of a TensorFlow graph; objektinitialisierung
            graph_def.ParseFromString(
                f.read()
            )  # Parse serialized protocol buffer data into variable
            _ = tf.import_graph_def(
                graph_def, name=""
            )  # import a serialized TensorFlow GraphDef protocol buffer, extract objects in the GraphDef as tf.Tensor

            # https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/inception.py ; ab zeile 276

        with tf.compat.v1.Session() as sess:
            softmax_tensor = sess.graph.get_tensor_by_name("final_result:0")
            # return: Tensor("final_result:0", shape=(?, 4), dtype=float32); stringname definiert in retrain.py, zeile 1064

            predictions = sess.run(
                softmax_tensor, {"DecodeJpeg/contents:0": image_data}
            )
            # gibt prediction values in array zuerueck:

            top_k = predictions[0].argsort()[-len(predictions[0]) :][::-1]
            # sortierung; circle -> 0, plus -> 1, square -> 2, triangle -> 3; array return bsp [3 1 2 0] -> sortiert nach groesster uebereinstimmmung

            # output
            result = {}
            for node_id in top_k:
                human_string = label_lines[node_id]
                score = predictions[0][node_id]
                result[human_string] = score
            score = max(result, key=lambda x: result[x])
            print(result)
