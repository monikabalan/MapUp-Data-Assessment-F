import pandas as pd
from scipy.spatial.distance import pdist, squareform


def calculate_distance_matrix(df):
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    numeric_cols = df.select_dtypes(include='number')
    distances = pdist(numeric_cols, metric='euclidean')
    dist_matrix = pd.DataFrame(squareform(distances), index=numeric_cols.index, columns=numeric_cols.index)

    return disk_matrix


def unroll_distance_matrix(df):
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    if df.shape[0] != df.shape[1]:
        raise ValueError("Input matrix is not square")
        ids = df.index.tolist()

    # Initialize lists to store unrolled data
    id_start = []
    id_end = []
    distance = []
 for i in range(len(ids)):
        for j in range(len(ids)):
            id_start.append(ids[i])
            id_end.append(ids[j])
            distance.append(df.iloc[i, j])
            unrolled_df = pd.DataFrame({'id_start': id_start, 'id_end': id_end, 'distance': distance})

    return unrolled_df


def find_ids_within_ten_percentage_threshold(df, reference_id):
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    avg_distances = df.groupby('id_start')['distance'].mean()
    ref_avg_distance = avg_distances[reference_id]
    threshold = ref_avg_distance * 0.1
    ids_within_threshold = avg_distances[
        (avg_distances >= ref_avg_distance - threshold) & 
        (avg_distances <= ref_avg_distance + threshold)
    ].reset_index()

    return ids_within_threshold


def calculate_toll_rate(df):
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    toll_rates = {
        'car': 0.1,    # Example rate for a car
        'truck': 0.5,  # Example rate for a truck
        'bus': 0.7     # Example rate for a bus
    } 
    df['toll_rate'] = df.apply(lambda row: toll_rates.get(row['vehicle_type'], 0) * row['distance'], axis=1)

 return df


def calculate_time_based_toll_rates(df):
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    time_intervals = {
        (0, 6): 0.1,    
        (6, 12): 0.2,   
        (12, 18): 0.3,  
        (18, 24): 0.4   
    }
    def get_time_based_rate(timestamp):
        hour = timestamp.hour
        for interval, rate in time_intervals.items():
            if interval[0] <= hour < interval[1]:
                return rate
        return 0  # Default rate if not found in any interval
     df['time_based_toll_rate'] = df['timestamp'].apply(get_time_based_rate)

    return df
