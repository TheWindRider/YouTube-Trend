import numpy
import pandas
import text_helpers
from tensorflow.python.platform import gfile
from sklearn.model_selection import train_test_split
from global_params import Params

def video_label(): 
    # List of category IDs as lookup in one direction
    video_english = pandas.read_csv(Params.DIR_DATA + Params.FILE_NAME_VIDEO)
    label_category = list(set(video_english["category_id"]))
    # Reverse it to a dict as lookup in the other direction
    category_label = {}
    for i in range(len(label_category)): 
        category_label[label_category[i]] = i    
    return label_category, category_label

def generate_data(): 
    # Load video data and transform text to numeric data
    video_file = Params.DIR_DATA + Params.FILE_NAME_VIDEO
    video_english = pandas.read_csv(video_file)
    video_english["words"] = text_helpers.text_to_numeric(video_english["title"])
    # Split to training and validation
    id_train, id_valid, X_train, X_valid, y_train, y_valid = train_test_split(
        video_english["video_id"], video_english["words"], video_english["category_id"], 
        test_size=Params.SPLIT_RATE)
    # Save to numpy file in the key-value format for each variable
    numpy.savez(Params.DIR_DATA + Params.FILE_NAME_MODEL, 
                id_train=id_train, id_valid=id_valid, 
                feature_train=X_train, feature_valid=X_valid, 
                label_train=y_train, label_valid=y_valid)

def load_data(): 
    """
    load features, load and transform labels
    they have been splitted previously into training and validation sets
    """
    # Load data from file
    word_english = pandas.read_csv(Params.DIR_DATA + Params.FILE_NAME_VOCAB, sep="\t", header=None)
    video_data = numpy.load(Params.DIR_DATA + Params.FILE_NAME_MODEL)
    # Transform label into range (0, NUM_CLASS)
    label_original, label_transform = video_label()
    # Cast to numpy arrays
    id_train = video_data["id_train"]
    id_valid = video_data["id_valid"]
    X_train = numpy.array(list(video_data["feature_train"]))
    X_valid = numpy.array(list(video_data["feature_valid"]))
    y_train = numpy.array([label_transform[label] for label in video_data["label_train"]])
    y_valid = numpy.array([label_transform[label] for label in video_data["label_valid"]])
    
    return id_train, id_valid, X_train, X_valid, y_train, y_valid, len(word_english)
    
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = numpy.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = numpy.random.permutation(numpy.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index: end_index]