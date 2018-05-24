import numpy
import pandas
from global_params import Params

def load_data(): 
    """
    load features, load and transform labels
    they have been splitted previously into training and validation sets
    """
    # Load data from file
    word_english = pandas.read_csv(Params.FILE_NAME_VOCAB, sep="\t", header=None)
    video_data = numpy.load(Params.FOLDER_NAME + Params.FILE_NAME_MODEL)
    # Transform label into range (0, NUM_CLASS)
    video_label = list(set(video_data["label_train"]))
    label_transform = {}
    for i in range(len(video_label)): 
        label_transform[video_label[i]] = i
    # Cast to numpy arrays
    X_train = numpy.array(list(video_data["feature_train"]))
    X_valid = numpy.array(list(video_data["feature_valid"]))
    y_train = numpy.array([label_transform[label] for label in video_data["label_train"]])
    y_valid = numpy.array([label_transform[label] for label in video_data["label_valid"]])
    
    return X_train, X_valid, y_train, y_valid, len(word_english)
    
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