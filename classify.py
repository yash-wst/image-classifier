import sys
import os
import pathlib
import argparse
import json


# speicherorte fuer trainierten graph und labels in train.sh festlegen ##

# Disable tensorflow compilation warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow.python.util import deprecation

OUTPUT_PATH = "output.json"
FLAGS = None
BASE_PATH = ""
DATA_DIR = ""


def process_dir(dir):
    """Method to read images from the
    given directory and classify them.
    Output is stored in the corresponding
    output file.

    Args:
        dir (string): Path to input directory
    """
    label_lines = [
        line.rstrip()
        for line in tf.compat.v1.gfile.GFile(
            os.path.join(DATA_DIR, "retrained_labels.txt")
        )
    ]

    with tf.compat.v1.gfile.FastGFile(
        os.path.join(DATA_DIR, "retrained_graph.pb"), "rb"
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

        with tf.compat.v1.Session() as sess:
            softmax_tensor = sess.graph.get_tensor_by_name("final_result:0")
            with open(FLAGS.output, "w", encoding="utf-8") as out_file:
                for img in os.listdir(dir):
                    image_data = tf.compat.v1.gfile.FastGFile(
                        f"{dir}/{img}", "rb"
                    ).read()
                    predictions = sess.run(
                        softmax_tensor, {"DecodeJpeg/contents:0": image_data}
                    )

                    top_k = predictions[0].argsort()[-len(predictions[0]) :][::-1]

                    # output
                    result = {"Image": f"{dir}/{img}"}
                    for node_id in top_k:
                        human_string = label_lines[node_id]
                        score = predictions[0][node_id]
                        result[human_string] = score

                    print(result)
                    json.dump(str(result), out_file)
                    out_file.write(",\n")


def process_image(img):
    """Method to read a single image
    and classify it. Output is stored in
    corresponding output file.

    Args:
        img (string): Path to image file
    """
    image_data = tf.compat.v1.gfile.FastGFile(img, "rb").read()

    # holt labels aus file in array
    label_lines = [
        line.rstrip()
        for line in tf.compat.v1.gfile.GFile(
            os.path.join(DATA_DIR, "retrained_labels.txt")
        )
    ]
    # !! labels befinden sich jeweils in eigenen lines -> keine aenderung in retrain.py noetig -> falsche darstellung im windows editor !!

    # graph einlesen, wurde in train.sh -> call retrain.py trainiert
    with tf.compat.v1.gfile.FastGFile(
        os.path.join(DATA_DIR, "retrained_graph.pb"), "rb"
    ) as f:
        graph_def = tf.compat.v1.GraphDef()
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

        predictions = sess.run(softmax_tensor, {"DecodeJpeg/contents:0": image_data})
        # gibt prediction values in array zuerueck:

        top_k = predictions[0].argsort()[-len(predictions[0]) :][::-1]
        # sortierung; circle -> 0, plus -> 1, square -> 2, triangle -> 3; array return bsp [3 1 2 0] -> sortiert nach groesster uebereinstimmmung

        # output
        result = {"Image": f"{img}"}
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            result[human_string] = score
        # score = max(result, key=lambda x: result[x])
        print(result)

        with open(FLAGS.output, "w", encoding="utf-8") as out_file:
            json.dump(str(result), out_file)
            out_file.write(",\n")


def process_file(file):
    """Method to read paths from
    the given input file and process
    each image for classification. Output
    is stored in corresponding output file.

    Args:
        file (string): Input txt file with paths
    """
    label_lines = [
        line.rstrip()
        for line in tf.compat.v1.gfile.GFile(
            os.path.join(DATA_DIR, "retrained_labels.txt")
        )
    ]

    with tf.compat.v1.gfile.FastGFile(
        os.path.join(DATA_DIR, "retrained_graph.pb"), "rb"
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

        with tf.compat.v1.Session() as sess:
            softmax_tensor = sess.graph.get_tensor_by_name("final_result:0")

            with open(file, "r") as input:
                with open(FLAGS.output, "w", encoding="utf-8") as out_file:
                    for img in input.readlines():
                        img = pathlib.Path(img.strip())
                        image_data = tf.compat.v1.gfile.FastGFile(f"{img}", "rb").read()
                        predictions = sess.run(
                            softmax_tensor, {"DecodeJpeg/contents:0": image_data}
                        )

                        top_k = predictions[0].argsort()[-len(predictions[0]) :][::-1]

                        # output
                        result = {"Image": f"{img}"}
                        for node_id in top_k:
                            human_string = label_lines[node_id]
                            score = predictions[0][node_id]
                            result[human_string] = score

                        print(result)
                        json.dump(str(result), out_file)
                        out_file.write("\n")


if __name__ == "__main__":
    deprecation._PRINT_DEPRECATION_WARNINGS = False

    # Supress deprecation warnings
    os.environ["AUTOGRAPH_VERBOSITY"] = "0"

    BASE_PATH = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_PATH, "tf_files")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        default="",
        help="Input .txt file with path of all the images",
    )
    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        default="",
        help="Path to image directory",
    )
    parser.add_argument(
        "-i",
        "--image",
        type=str,
        default="",
        help="Single image file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=OUTPUT_PATH,
        help="Output file",
    )

    FLAGS, unparsed = parser.parse_known_args()

    try:
        if FLAGS.directory:
            process_dir(FLAGS.directory)

        if FLAGS.image:
            process_image(FLAGS.image)

        if FLAGS.file:
            process_file(FLAGS.file)

    except Exception as e:
        print(f"Encountered exception: {e}")
