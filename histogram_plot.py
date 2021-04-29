import numpy as np
import matplotlib.pyplot as plt

def load_dataset():
    sample_label_pairs = [] # [ (sample, label) ]

    with open(f"iris.data", 'rb') as csv_file:
        for row in csv_file:
            cells = [ cell.strip() for cell in row.decode().split(',') ]

            label = cells[ len(cells)-1 ]
            sample = np.array(cells[ :len(cells)-1], dtype=np.float32)

            sample_label_pairs.append( (sample, label) )

    sample_label_pairs = np.array(sample_label_pairs)
    samples, labels = np.transpose(sample_label_pairs)

    return samples, labels

# Map indices of features to human friendly names:
def get_all_features():
    return {0: 'Sepal length',
            1: 'Sepal width',
            2: 'Petal length',
            3: 'Petal width'}

def plot_histograms(samples, labels, features, step_length=0.1):
    classes = np.unique(labels)

    num_subplots = len(features)
    num_cols = np.uint8(np.floor(np.sqrt(num_subplots)))
    num_rows = np.uint8(np.ceil(num_subplots / num_cols))

    fig = plt.figure(figsize=(7, 7))

    for feature_index in range(len(features)):
        feature = features[feature_index]

        ax = fig.add_subplot(num_cols, num_rows, feature_index+1)
        ax.set(xlabel='Measurement[cm]', ylabel='Number of samples')

        for curr_class in classes:
            sample_indices = np.where(labels == curr_class)[0]

            samples_matching_class = [ samples[i] for i in sample_indices ]
            measurements_by_features = np.transpose(samples_matching_class)
            if measurements_by_features.any():
                measurements_matching_feature = measurements_by_features[feature_index]

                ax.hist(measurements_matching_feature, alpha=0.5, stacked=True, label=curr_class)

        ax.set_title(feature)
        ax.legend(prop={'size': 7})

    plt.show()

def show_histograms():

    all_samples, all_labels = load_dataset()
    features = get_all_features()

    plot_histograms(all_samples, all_labels, features)

show_histograms()