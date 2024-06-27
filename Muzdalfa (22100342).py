from sklearn.metrics import silhouette_score
import sklearn.cluster as cluster
import cluster_tools as ct
import errors as err
import scipy.optimize as opt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
os.environ["OMP_NUM_THREADS"] = '1'

pd.options.mode.chained_assignment = None


def load_csv_data(file_path):
    """
    Load a CSV file into a pandas DataFrame, skipping the first four rows and
    dropping certain columns.

    Parameters:
        file_path (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: Data loaded from the CSV.
    """
    print(file_path)
    data = pd.read_csv(file_path, skiprows=4)
    data = data.drop(
        columns=[
            'Country Code',
            'Indicator Name',
            'Indicator Code',
            'Unnamed: 67'])
    return data


def prepare_clustering_data(
        indicator_1_name,
        indicator_2_name,
        df1,
        df2,
        year):
    """
    Prepare data for clustering based on two indicators for a specific year.

    Parameters
    ----------
    indicator_1_name : str
        First Indicator Name.
    indicator_2_name : str
        Second Indicator Name.
    df1 : pd.DataFrame
        First DataFrame.
    df2 : pd.DataFrame
        Second DataFrame.
    year : str
        Year for which the data is required.

    Returns
    -------
    df_cluster : pd.DataFrame
        DataFrame for clustering.
    """
    df1 = df1[['Country Name', year]]
    df2 = df2[['Country Name', year]]
    merged_df = pd.merge(df1, df2, on="Country Name", how="outer")
    merged_df = merged_df.dropna()
    merged_df = merged_df.rename(
        columns={
            year +
            "_x": indicator_1_name,
            year +
            "_y": indicator_2_name})
    df_cluster = merged_df[[indicator_1_name, indicator_2_name]].copy()
    return df_cluster


def merge_indicator_datasets(
        indicator_1_name,
        indicator_2_name,
        df1,
        df2,
        year):
    """
    Merge two datasets based on the country name and specific year.

    Parameters:
        indicator_1_name (str): Name of the first indicator.
        indicator_2_name (str): Name of the second indicator.
        df1 (pd.DataFrame): First dataset.
        df2 (pd.DataFrame): Second dataset.
        year (str): Year of interest.

    Returns:
        pd.DataFrame: Merged data with two indicators.
    """
    df1 = df1[['Country Name', year]]
    df2 = df2[['Country Name', year]]
    merged_data = pd.merge(df1, df2, on="Country Name", how="outer").dropna()
    merged_data = merged_data.rename(
        columns={
            year + "_x": indicator_1_name,
            year + "_y": indicator_2_name})
    return merged_data[['Country Name', indicator_1_name, indicator_2_name]]


def logistic_growth_model(t, start, rate, midpoint):
    """
    Logistic growth model function.

    Parameters:
        t (array-like): Input times or years.
        start (float): The initial population or value.
        rate (float): The growth rate.
        midpoint (float): The midpoint of growth (inflection point).

    Returns:
        array-like: Modeled logistic growth values.
    """
    return start / (1.0 + np.exp(-rate * (t - midpoint)))


def fit_and_forecast_growth(
        data,
        country,
        indicator,
        fit_title,
        forecast_title,
        initial_params):
    """
    Fit data to a logistic model and forecast future values.

    Parameters:
        data (pd.DataFrame): Data for fitting.
        country (str): Country name.
        indicator (str): Indicator name.
        fit_title (str): Title for the fit plot.
        forecast_title (str): Title for the forecast plot.
        initial_params (tuple): Initial parameters for the logistic model fit.

    Returns:
        None: This function plots the results.
    """
    popt, pcov = opt.curve_fit(
        logistic_growth_model, data.index, data[country], p0=initial_params)
    data["Fitted Data"] = logistic_growth_model(data.index, *popt)

    plt.figure()
    plt.plot(
        data.index,
        data[country],
        color='blue',
        linestyle='-',
        label="Data")
    plt.plot(
        data.index,
        data["Fitted Data"],
        color='green',
        linestyle='--',
        label="Fit")
    plt.legend()
    plt.xlabel('Year')
    plt.ylabel(indicator)
    plt.title(fit_title)
    plt.savefig(f'{country}_fit.png', dpi=300)

    future_years = np.linspace(1995, 2030)
    future_fit = logistic_growth_model(future_years, *popt)
    error_bounds = err.error_prop(
        future_years, logistic_growth_model, popt, pcov)

    plt.figure()
    plt.plot(
        data.index,
        data[country],
        color='blue',
        linestyle='-',
        label="Data")
    plt.plot(
        future_years,
        future_fit,
        color='green',
        linestyle='--',
        label="Forecast")
    plt.fill_between(
        future_years,
        future_fit -
        error_bounds,
        future_fit +
        error_bounds,
        color="yellow",
        alpha=0.5)
    plt.legend(loc="upper left")
    plt.xlabel('Year')
    plt.ylabel(indicator)
    plt.title(forecast_title)
    plt.savefig(f'{country}_forecast.png', dpi=300)
    plt.show()


def extract_country_specific_data(dataset, country, start_year, end_year):
    """
    Extract and filter data for a specific country and time range.

    Parameters:
        dataset (pd.DataFrame): The dataset containing country data.
        country (str): Country name.
        start_year (int): Start year.
        end_year (int): End year.

    Returns:
        pd.DataFrame: Filtered country data.
    """
    dataset = dataset.T
    dataset.columns = dataset.iloc[0]
    dataset = dataset.drop(['Country Name'])
    dataset = dataset[[country]]
    dataset.index = dataset.index.astype(int)
    dataset = dataset[(dataset.index > start_year) &
                      (dataset.index <= end_year)]
    dataset[country] = dataset[country].astype(float)
    return dataset


def plot_data_clusters(
        data,
        x_label,
        y_label,
        plot_title,
        num_clusters,
        scaled_data,
        data_min,
        data_max):
    """
    Plot data clusters using KMeans clustering.

    Parameters:
        data (pd.DataFrame): Original data.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        plot_title (str): Title for the graph.
        num_clusters (int): Number of clusters to form.
        scaled_data (pd.DataFrame): Scaled data for clustering.
        data_min (float): Minimum value for scaling.
        data_max (float): Maximum value for scaling.

    Returns:
        np.ndarray: Cluster labels.
    """
    kmeans = cluster.KMeans(n_clusters=num_clusters, n_init=10, random_state=0)
    kmeans.fit(scaled_data)
    labels = kmeans.labels_
    centroids = ct.backscale(kmeans.cluster_centers_, data_min, data_max)
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(data[x_label], data[y_label], c=labels, cmap="tab10")
    plt.scatter(centroids[:, 0], centroids[:, 1], c="black", marker="d", s=80)
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(plot_title)
    plt.savefig('cluster_plot.png', dpi=300)
    plt.show()
    return labels


def evaluate_silhouette_scores(data, max_clusters=10):
    """
    Evaluate and plot silhouette scores for different numbers of clusters.

    Parameters:
        data (pd.DataFrame): Data for clustering.
        max_clusters (int): Maximum number of clusters to evaluate.

    Returns:
        None: This function plots the silhouette scores.
    """
    scores = []
    for clusters in range(2, max_clusters + 1):
        kmeans = cluster.KMeans(
            n_clusters=clusters,
            random_state=42,
            n_init=10)
        labels = kmeans.fit_predict(data)
        score = silhouette_score(data, labels)
        scores.append(score)
    plt.figure(figsize=(8, 6))
    plt.plot(range(2, max_clusters + 1), scores, 'g-o')
    plt.title('Silhouette Scores for Varying Cluster Counts')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.show()


# Loading the data
co2_data = load_csv_data('CO2_emissions_metric_tons_per_capita.csv')
UP_data = load_csv_data('Urban_population _%_of_total_population.csv')

# Merging the indicators into a single DataFrame for clustering
clustering_data = prepare_clustering_data(
    'Urban Population Percentage',
    'CO2 Emissions (Metric Tons per Capita)',
    UP_data,
    co2_data,
    '2020')

# Scaling data and evaluating silhouette scores
scaled_clustering_data, min_vals, max_vals = ct.scaler(clustering_data)
evaluate_silhouette_scores(scaled_clustering_data, 12)

# Plotting clusters and updating the DataFrame with cluster labels
cluster_labels = plot_data_clusters(
    clustering_data,
    'Urban Population Percentage',
    'CO2 Emissions (Metric Tons per Capita)',
    'CO2 Emissions vs Urban Population Percentage in 2020',
    2,
    scaled_clustering_data,
    min_vals,
    max_vals)

# Merging datasets for additional analysis
merged_data = merge_indicator_datasets(
    'Urban Population Percentage',
    'CO2 Emissions (Metric Tons per Capita)',
    UP_data,
    co2_data,
    '2020')
merged_data['Cluster Label'] = cluster_labels

# Filtering data for specific countries
specific_countries_data = merged_data[merged_data['Country Name'].isin(
    ['United States', 'China'])]

# Extracting, fitting, and predicting for China
china_UP_data = extract_country_specific_data(UP_data, 'China', 1990, 2020)
china_UP_data = china_UP_data.fillna(0)
fit_and_forecast_growth(
    china_UP_data,
    'China',
    'Urban Population Percentage',
    "Urban Population Percentage in China 1990-2020",
    "Urban Population Percentage Forecast for China Until 2030",
    (1e5,
     0.02,
     1990))

# Extracting, fitting, and predicting for India
US_UP_data = extract_country_specific_data(
    UP_data, 'United States', 1990, 2020)
US_UP_data = US_UP_data.fillna(0)
fit_and_forecast_growth(
    US_UP_data,
    'United States',
    'Urban Population Percentage',
    "Urban Population Percentage in United States 1990-2020",
    "Urban Population Percentage Forecast for United States Until 2030",
    (1e5,
     0.04,
     1990))

# Extracting, fitting, and predicting for India
US_co2_data = extract_country_specific_data(
    co2_data, 'United States', 1990, 2020)
US_co2_data = US_co2_data.fillna(0)
fit_and_forecast_growth(
    US_co2_data,
    'United States',
    'CO2 Emissions (Metric Tons per Capita)',
    "CO2 Emissions in United States 1990-2020",
    "CO2 Emissions Forecast for United States Until 2030",
    (1e5,
     0.04,
     1990))

# Extracting, fitting, and predicting for China
china_co2_data = extract_country_specific_data(co2_data, 'China', 1990, 2020)
china_co2_data = china_co2_data.fillna(0)
fit_and_forecast_growth(
    china_co2_data,
    'China',
    'CO2 Emissions (Metric Tons per Capita)',
    "CO2 Emissions in China 1990-2020",
    "CO2 Emissions Forecast for China Until 2030",
    (1e5,
     0.02,
     1990))
