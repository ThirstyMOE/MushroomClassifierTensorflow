# import math
from IPython import display
# from matplotlib import cm
# from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)  # Set tensorflow logging verbosity to: Errors
pd.options.display.max_rows = 10  # Set the Pandas dataframe display rows to a max of 10 at a time
pd.options.display.float_format = '{:.1f}'.format  # Set the Pandas dataframe display decimals to 1 place.

# HYPERPARAMETERS
periods = 7
learning_rate = 0.5  # 1.0
regularization_strength = 5.0  # 5.0
steps = 300  # 300
batch_size = 100  # 100

# ORIGINAL DATA SET FROM KAGGLE
# https://www.kaggle.com/uciml/mushroom-classification/data

# Read in the raw dataset
mushroom_dataframe = pd.read_csv(
    "C:/Users/eyoon/PycharmProjects/GoogleMachineLearningDNN/mushroom-classification/mushrooms.csv", sep=",")

mushroom_dataframe = mushroom_dataframe.reindex(  # change the Pandas dataframe's ordering indices
    np.random.permutation(mushroom_dataframe.index))  # create a random permutation off of the dataframe's indices

# print(mushroom_dataframe.describe())  # Pandas describes the dataset to us


def preprocess_features(mushroom_dataframe):
    """Prepares input features from mushrooms data set.

    Args:
    california_housing_dataframe: A Pandas DataFrame expected to contain data
      from the California housing data set.
    Returns:
    A DataFrame that contains the features to be used for the model, including
    synthetic features.
    """
    selected_features = mushroom_dataframe[
        ["cap-shape",
         "cap-surface",
         "cap-color",
         "bruises",
         "odor",
         "gill-attachment",
         "gill-spacing",
         "gill-size",
         "gill-color",
         "stalk-shape",
         "stalk-root",
         "stalk-surface-above-ring",
         "stalk-surface-below-ring",
         "stalk-color-above-ring",
         "stalk-color-below-ring",
         "veil-type",
         "veil-color",
         "ring-number",
         "ring-type",
         "spore-print-color",
         "population",
         "habitat"]
        ]

    processed_features = selected_features.copy()

    # No synthetic features yet

    return processed_features


def preprocess_targets(mushroom_dataframe):
    """Prepares target features (i.e., labels) from mushroom data set.

    Args:
    california_housing_dataframe: A Pandas DataFrame expected to contain data
      from the California housing data set.
    Returns:
    A DataFrame that contains the target feature.
    """
    output_targets = pd.DataFrame()
    # Create a boolean categorical feature representing whether the
    # class is poisonous or not.
    output_targets["is_mushroom_poisonous"] = (
            mushroom_dataframe["class"] == "p")
    return output_targets


# Choose the first 5734 (out of 8124) examples for training.
training_examples = preprocess_features(mushroom_dataframe.head(5734))
training_targets = preprocess_targets(mushroom_dataframe.head(5734))

# Choose the last 2390 (out of 8124) examples for validation.
validation_examples = preprocess_features(mushroom_dataframe.tail(2390))
validation_targets = preprocess_targets(mushroom_dataframe.tail(2390))

# Double-check that we've done the right thing.
print("Training examples summary:")
display.display(training_examples.describe())
print("Validation examples summary:")
display.display(validation_examples.describe())

print("Training targets summary:")
display.display(training_targets.describe())
print("Validation targets summary:")
display.display(validation_targets.describe())


def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model.

    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """

    # Convert pandas data into a dict of np arrays.
    features = {key: np.array(value) for key, value in dict(features).items()}

    # Construct a data set, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features, targets))  # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle the data, if specified.
    if shuffle:
        ds = ds.shuffle(5734)  # TODO: This number??

    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


def get_quantile_based_buckets(feature_values, num_buckets):
    quantiles = feature_values.quantile(
        [(i+1.)/(num_buckets + 1.) for i in range(num_buckets)])
    return [quantiles[q] for q in quantiles.keys()]


def construct_feature_columns():
    """Construct the TensorFlow Feature Columns.

    Returns:
      A set of feature columns
    """

    # Link for categorical data encoding
    # https://www.tensorflow.org/tutorials/wide

    cap_shape = tf.feature_column.categorical_column_with_vocabulary_list(
        "cap-shape", ["b", "c", "x", "f", "k", "s"]
    )
    cap_surface = tf.feature_column.categorical_column_with_vocabulary_list(
        "cap-surface", ["f", "g", "y", "s"]
    )
    cap_color = tf.feature_column.categorical_column_with_vocabulary_list(
        "cap-color", ["n", "b", "c", "g", "r", "p", "u", "e", "w", "y"]
    )
    bruises = tf.feature_column.categorical_column_with_vocabulary_list(
        "bruises", ["t", "f"]
    )
    odor = tf.feature_column.categorical_column_with_vocabulary_list(
        "odor", ["a", "l", "c", "y", "f", "m", "n", "p", "s"]
    )
    gill_attachment = tf.feature_column.categorical_column_with_vocabulary_list(
        "gill-attachment", ["a", "d", "f", "n"]
    )
    gill_spacing = tf.feature_column.categorical_column_with_vocabulary_list(
        "gill-spacing", ["c", "w", "d"]
    )
    gill_size = tf.feature_column.categorical_column_with_vocabulary_list(
        "gill-size", ["b", "n"]
    )
    gill_color = tf.feature_column.categorical_column_with_vocabulary_list(
        "gill-color", ["k", "n", "b", "h", "g"]
    )
    stalk_shape = tf.feature_column.categorical_column_with_vocabulary_list(
        "stalk-shape", ["e", "t"]
    )
    stalk_root = tf.feature_column.categorical_column_with_vocabulary_list(
        "stalk-root", ["b", "c", "u", "e", "z", "r", "?"]
    )
    stalk_surface_above_ring = tf.feature_column.categorical_column_with_vocabulary_list(
        "stalk-surface-above-ring", ["f", "y", "k", "s"]
    )
    stalk_surface_below_ring = tf.feature_column.categorical_column_with_vocabulary_list(
        "stalk-surface-below-ring", ["f", "y", "k", "s"]
    )
    stalk_color_above_ring = tf.feature_column.categorical_column_with_vocabulary_list(
        "stalk-color-above-ring", ["n", "b", "c", "g", "o", "p", "e", "w", "y"]
    )
    stalk_color_below_ring = tf.feature_column.categorical_column_with_vocabulary_list(
        "stalk-color-below-ring", ["n", "b", "c", "g", "o", "p", "e", "w", "y"]
    )
    veil_type = tf.feature_column.categorical_column_with_vocabulary_list(
        "veil-type", ["p", "u"]
    )
    veil_color = tf.feature_column.categorical_column_with_vocabulary_list(
        "veil-color", ["n", "o", "w", "y"]
    )
    ring_number = tf.feature_column.categorical_column_with_vocabulary_list(
        "ring-number", ["n", "o", "t"]
    )
    ring_type = tf.feature_column.categorical_column_with_vocabulary_list(
        "ring-type", ["c", "e", "f", "l", "n", "p", "s", "z"]
    )
    spore_print_color = tf.feature_column.categorical_column_with_vocabulary_list(
        "spore-print-color", ["k", "n", "b", "h", "r", "o", "u", "w", "y"]
    )
    population = tf.feature_column.categorical_column_with_vocabulary_list(
        "population", ["a", "c", "n", "s", "v", "y"]
    )
    habitat = tf.feature_column.categorical_column_with_vocabulary_list(
        "habitat", ["g", "l", "m", "p", "u", "w", "d"]
    )
    # feature_columns = set([
    #     cap_shape,
    #     cap_surface,
    #     cap_color,
    #     # bruises,
    #     odor,
    #     # gill_attachment,
    #     gill_spacing,
    #     gill_size,
    #     # gill_color,
    #     stalk_shape,
    #     stalk_root,
    #     # stalk_surface_above_ring,
    #     # stalk_surface_below_ring,
    #     stalk_color_above_ring,
    #     stalk_color_below_ring,
    #     # veil_type,
    #     # veil_color,
    #     # ring_number,
    #     # ring_type,
    #     spore_print_color,
    #     population,
    #     habitat
    # ])

    # long_x_lat = tf.feature_column.crossed_column(
    #     set([cap_shape, cap_surface]), hash_bucket_size=1000)
    feature_columns = {
        cap_shape,
        cap_surface,
        cap_color,
        # bruises,
        odor,
        # gill_attachment,
        # gill_spacing,
        # gill_size,
        # gill_color,
        # stalk_shape,
        # stalk_root,
        # stalk_surface_above_ring,
        # stalk_surface_below_ring,
        # stalk_color_above_ring,
        # stalk_color_below_ring,
        # veil_type,
        # veil_color,
        # ring_number,
        # ring_type,
        # spore_print_color,
        # population,
        # habitat
    }

    return feature_columns


def model_size(estimator):
    variables = estimator.get_variable_names()
    size = 0
    for variable in variables:
        if not any(x in variable for x in ['global_step', 'centered_bias_weight', 'bias_weight', 'Ftrl']):
            size += np.count_nonzero(estimator.get_variable_value(variable))
    return size


def train_linear_classifier_model(
        learning_rate,
        regularization_strength,
        steps,
        batch_size,
        feature_columns,
        training_examples,
        training_targets,
        validation_examples,
        validation_targets):
    """Trains a linear regression model.

    In addition to training, this function also prints training progress information,
    as well as a plot of the training and validation loss over time.

    Args:
      learning_rate: A `float`, the learning rate.
      regularization_strength: A `float` that indicates the strength of the L1
         regularization. A value of `0.0` means no regularization.
      steps: A non-zero `int`, the total number of training steps. A training step
        consists of a forward and backward pass using a single batch.
      feature_columns: A `set` specifying the input feature columns to use.
      training_examples: A `DataFrame` containing one or more columns from
        `california_housing_dataframe` to use as input features for training.
      training_targets: A `DataFrame` containing exactly one column from
        `california_housing_dataframe` to use as target for training.
      validation_examples: A `DataFrame` containing one or more columns from
        `california_housing_dataframe` to use as input features for validation.
      validation_targets: A `DataFrame` containing exactly one column from
        `california_housing_dataframe` to use as target for validation.

    Returns:
      A `LinearClassifier` object trained on the training data.
    """

    steps_per_period = steps / periods

    # Create a linear classifier object.
    my_optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate,
                                          l1_regularization_strength=regularization_strength)
    # my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_classifier = tf.estimator.LinearClassifier(
        feature_columns=feature_columns,
        optimizer=my_optimizer
    )

    # Create input functions.
    training_input_fn = lambda: my_input_fn(training_examples,
                                            training_targets["is_mushroom_poisonous"],
                                            batch_size=batch_size)
    predict_training_input_fn = lambda: my_input_fn(training_examples,
                                                    training_targets["is_mushroom_poisonous"],
                                                    num_epochs=1,
                                                    shuffle=False)
    predict_validation_input_fn = lambda: my_input_fn(validation_examples,
                                                      validation_targets["is_mushroom_poisonous"],
                                                      num_epochs=1,
                                                      shuffle=False)

    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    print("Training model...")
    print("LogLoss (on validation data):")
    training_log_losses = []
    validation_log_losses = []
    for period in range(0, periods):
        # Train the model, starting from the prior state.
        linear_classifier.train(
            input_fn=training_input_fn,
            steps=steps_per_period
        )
        # Take a break and compute predictions.
        training_probabilities = linear_classifier.predict(input_fn=predict_training_input_fn)
        training_probabilities = np.array([item['probabilities'] for item in training_probabilities])

        validation_probabilities = linear_classifier.predict(input_fn=predict_validation_input_fn)
        validation_probabilities = np.array([item['probabilities'] for item in validation_probabilities])

        # Compute training and validation loss.
        training_log_loss = metrics.log_loss(training_targets, training_probabilities)
        validation_log_loss = metrics.log_loss(validation_targets, validation_probabilities)
        # Occasionally print the current loss.
        print("  period %02d : %0.3f" % (period, validation_log_loss))
        # Add the loss metrics from this period to our list.
        training_log_losses.append(training_log_loss)
        validation_log_losses.append(validation_log_loss)
    print("Model training finished.")

    # Output a graph of loss metrics over periods.
    plt.ylabel("LogLoss")
    plt.xlabel("Periods")
    plt.title("LogLoss vs. Periods")
    plt.tight_layout()
    plt.plot(training_log_losses, label="training")
    plt.plot(validation_log_losses, label="validation")
    plt.legend()
    plt.show()

    # Get AUC metrics
    evaluation_metrics = linear_classifier.evaluate(input_fn=predict_validation_input_fn)

    print("AUC on the validation set: %0.3f" % evaluation_metrics['auc'])
    print("Accuracy on the validation set: %0.3f" % evaluation_metrics['accuracy'])

    validation_probabilities = linear_classifier.predict(input_fn=predict_validation_input_fn)
    # Get just the probabilities for the positive class.
    validation_probabilities = np.array([item['probabilities'][1] for item in validation_probabilities])

    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(
        y_true=validation_targets, y_score=validation_probabilities)
    plt.plot(false_positive_rate, true_positive_rate, label="our model")
    plt.plot([0, 1], [0, 1], label="random classifier")
    _ = plt.legend(loc=2)
    plt.show()

    return linear_classifier


linear_classifier = train_linear_classifier_model(
    learning_rate=learning_rate,
    regularization_strength=regularization_strength,
    steps=steps,
    batch_size=batch_size,
    feature_columns=construct_feature_columns(),
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)

print("Model size:", model_size(linear_classifier))


