import numpy as np
import logging

# Logging
LOG_FORMAT = '%(levelname)s - %(asctime)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)
logger.info('Logging started')

def load_data(file_name):
    """Loads dataset from csv file to a numpy array"""
    dataset = np.genfromtxt(file_name, delimiter=',')
    return dataset

# Compute Pearson correlation coefficient between all pairs of items in the matrix.
def pearson_similarity(matrix):
    # Center each row by subtracting the row mean
    row_means = np.mean(matrix, axis=1, keepdims=True)
    centered_matrix = matrix - row_means
    
    # Compute the similarity
    # Calculate the dot product of the centered matrix with its transpose
    similarity_numerator = np.dot(centered_matrix, centered_matrix.T)
    
    # Calculate the normalization factor for the denominator
    sum_squared_diffs = np.sqrt(np.sum(centered_matrix**2, axis=1, keepdims=True))
    normalization_factor = np.dot(sum_squared_diffs, sum_squared_diffs.T)
    
    # Compute Pearson correlation coefficient matrix
    similarity_matrix = similarity_numerator / normalization_factor
    
    # Ensure the diagonal is 0 as we don't want to include self-similarity
    np.fill_diagonal(similarity_matrix, 0)
    
    return similarity_matrix

# Predict the rating a user might give to an item based on item similarity
def predict_rating(user_id, item_id, user_item_matrix, item_similarity, user_avg_ratings, k=55):
    # Locates the index of the specified user_id and item_id in their respective arrays
    user_index = np.where(user_ids == user_id)[0]
    item_index = np.where(item_ids == item_id)[0]
    # If either user_id or item_id is not found, log a warning and return a default rating of 0
    if not user_index.size or not item_index.size:
        logger.warning(f"User ID {user_id} or Item ID {item_id} not found.")
        return 0
    # Simplifies user_index and item_index to scalar values assuming unique identifiers
    user_index, item_index = user_index[0], item_index[0]

    # Extracts all similarity scores for the target item against all others
    similarities = item_similarity[item_index]
    # Sets the similarity of the item to itself to negative infinity to exclude it from consideration
    similarities[item_index] = -np.inf

    # Finds the indices of the top-k items with the highest similarity scores
    top_k_indices = np.argsort(similarities)[-k:]

    # Fetches the ratings given by the user to these top-k most similar items
    user_ratings = user_item_matrix[user_index, top_k_indices]
    # Extracts the corresponding similarity scores for these top-k items
    top_k_similarities = similarities[top_k_indices]

    # Identifies which of these top-k items have been rated by the user
    rated_indices = user_ratings > 0
    # If none of the top-k similar items have been rated by the user, use the user's average rating as a fallback
    if not np.any(rated_indices):
        return user_avg_ratings.get(user_id, 0)

    # Calculates the predicted rating as a weighted average, where weights are the item similarities
    filtered_similarities = top_k_similarities[rated_indices]
    filtered_ratings = user_ratings[rated_indices]
    # The predicted rating is the dot product of ratings and similarities, divided by the sum of similarities
    prediction = np.dot(filtered_ratings, filtered_similarities) / np.sum(filtered_similarities)
    
    return prediction

# Remove rating outliers for each item using the Z-score method
def remove_outliers(data):
    # Identifies unique item IDs
    item_ids = np.unique(data[:, 1])
    for item_id in item_ids:
        # Finds indices of all ratings for the current item
        indices = np.where(data[:, 1] == item_id)[0]
        item_ratings = data[indices, 2]
        # Calculates the mean and standard deviation of these ratings
        mean = np.mean(item_ratings)
        std = np.std(item_ratings)
        # Calculates Z-scores of ratings (how many standard deviations away a rating is from the mean)
        if std > 0:
            z_scores = np.abs((item_ratings - mean) / std)
            # Marks ratings as outliers if their Z-score is greater than 3
            outliers = z_scores > 3
            # Replaces outlier ratings with 0
            data[indices[outliers], 2] = 0
    return data

# Randomly split the dataset into training and testing sets based on a specified ratio (used for my local testing)
def train_test_split(data, train_ratio=0.8):
    # Shuffles the dataset in-place
    np.random.shuffle(data)
    # Calculates the size of the training set as a proportion of the total dataset
    train_size = int(len(data) * train_ratio)
    # Splits the dataset
    train_set = data[:train_size]
    test_set = data[train_size:]
    return train_set, test_set

# Calculates the Mean Absolute Error between predictions and actual target values (used for my local testing)
def calculate_mae(predictions, targets):
    # MAE is the average of the absolute differences between predictions and actual values
    return np.mean(np.abs(predictions - targets))

# Writes predictions to a file, each on a new line.
def produce_file(filename, predictions):
    with open(filename, 'w') as f:
        for row in predictions:
            # Converts each row of predictions into a comma-separated string and write to the file
            f.write(','.join(map(str, row)) + '\n')

# Creates a zero-initialized matrix for storing user-item ratings
def initialise_matrix(user_ids, item_ids):
    matrix = np.zeros((len(user_ids), len(item_ids)))
    return matrix

# Calculates average ratings given by each user.
def calculate_user_avg_ratings(train_data, user_ids):
    user_avg_ratings = {}
    for user_id in user_ids:
        # Extract all ratings made by the current user
        user_ratings = train_data[train_data[:, 0] == user_id, 2]
        # Calculate the mean of these ratings
        avg_rating = np.mean(user_ratings)
        # Store this mean rating in a dictionary with the user_id as the key
        user_avg_ratings[user_id] = avg_rating
    return user_avg_ratings

if __name__ == '__main__':
    # Main execution flow

    # Loads training and test datasets
    train_data = load_data('train_100k_withratings.csv')
    # Removes outliers from training data
    train_data = remove_outliers(train_data)
    # Identifies unique user and item IDs
    user_ids, item_ids = np.unique(train_data[:, 0]), np.unique(train_data[:, 1])

    # Calculates average ratings per user based on the training data
    user_avg_ratings = calculate_user_avg_ratings(train_data, user_ids)

    # Initializes the user-item matrix and populate it with the training data ratings
    user_item_matrix = initialise_matrix(user_ids, item_ids)
    for row in train_data:
        if len(row) >= 3:  # Ensures row has at least user ID, item ID, and rating
            user_id, item_id, rating = row[:3]
            user_index = np.where(user_ids == user_id)[0][0]
            item_index = np.where(item_ids == item_id)[0][0]
            user_item_matrix[user_index, item_index] = rating

    # Calculates item similarity matrix
    item_similarity = pearson_similarity(user_item_matrix.T)

    # Loads test data without ratings
    test_data = load_data('test_100k_withoutratings.csv')
    
    predictions = []
    for row in test_data:
        if len(row) >= 2:  # Ensures row has at least user ID and item ID
            user_id, item_id, timestamp = row[:3]
            # Predicts rating for each (user, item) pair in the test set
            prediction = predict_rating(user_id, item_id, user_item_matrix, item_similarity, user_avg_ratings)
            # Rounds the prediction to the nearest integer
            prediction = np.round(prediction)
            # Stores predictions in a list
            predictions.append([int(user_id), int(item_id), float(prediction), int(timestamp)])

    # Outputs predictions to a CSV file
    produce_file('submission.csv', predictions)
    
    logger.info("submission.csv produced")
