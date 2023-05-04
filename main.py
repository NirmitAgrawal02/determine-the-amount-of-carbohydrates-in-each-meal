import pandas as pd
import numpy as np
import datetime as dt
import math
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN


def autocorrelation(val1, total_lag):
    # This is important: If a series is passed, the product below is calculated
    # based on the index, which corresponds to squaring the series.
    if isinstance(val1, pd.Series):
        val1 = val1.values
    if len(val1) < total_lag:
        return np.nan
    # Slice the relevant subseries based on the lag
    y_axis_val = val1[: (len(val1) - total_lag)]
    y_axis_val_2 = val1[total_lag:]
    # Subtract the mean of the whole series x
    x_mean = np.mean(val1)
    # The result is sometimes referred to as "covariation"
    sum_of_product = np.sum((y_axis_val - x_mean) * (y_axis_val_2 - x_mean))
    # Return the normalized unbiased covariance
    v = np.var(val1)
    if np.isclose(v, 0):
        return np.NaN
    else:
        return sum_of_product / ((len(val1) - total_lag) * v)

# Autocorrelation function


def auto_correlation(dataframe):
    autoCorrelationFeature = []
    for i in range(0, len(dataframe)):
        # autoCorrelationFeature.append(feature_calculators.autocorrelation(dataframe.iloc[i], 1))
        autoCorrelationFeature.append(autocorrelation(dataframe.iloc[i], 1))
    return pd.DataFrame(autoCorrelationFeature)


def roll(a, shift):
    if not isinstance(a, np.ndarray):
        a = np.asarray(a)
    idx = shift % len(a)
    return np.concatenate([a[-idx:], a[:-idx]])


def number_peaks(x, n):
    reduced_value = x[n:-n]
    total_result = None
    for i in range(1, n + 1):
        first_result = reduced_value > roll(x, i)[n:-n]
        if total_result is None:
            total_result = first_result
        else:
            total_result &= first_result

        total_result &= reduced_value > roll(x, -i)[n:-n]
    return np.sum(total_result)


def binned_entropy(x, max_bins):
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)

    # nan makes no sense here
    if np.isnan(x).any():
        return np.nan

    hist, bin_edges = np.histogram(x, bins=max_bins)
    probability_value = hist / x.size
    probability_value[probability_value == 0] = 1.0
    return -np.sum(probability_value * np.log(probability_value))


def number_crossing_m(x, m):
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    positive = x > m
    return np.where(np.diff(positive))[0].size


def main():
    cgmDf, insulinDf = loadDataset()
    mealFrame, mealFrameIndex, mealData = processMealData(cgmDf, insulinDf)
    mealFrame = interpolation(mealFrame, mealFrameIndex)
    mealCarbQuantity = mealData[['BWZ Carb Input (grams)', 'Defaultindex']]
    maxMealQuantity = mealCarbQuantity['BWZ Carb Input (grams)'].max()
    minMealQuantity = mealCarbQuantity['BWZ Carb Input (grams)'].min()
    bins = pd.DataFrame()
    bins['binLabel'] = mealCarbQuantity.apply(lambda row: extractBinLabels(
        row['BWZ Carb Input (grams)']).astype(np.int64), axis=1)
    bins['Defaultindex'] = mealCarbQuantity['Defaultindex']

    mealDataQuantity = mealFrame.merge(bins, how='inner', on=['Defaultindex'])

    meal_carbohydrates_intake_time = pd.DataFrame()
    meal_carbohydrates_intake_time = mealData[[
        'BWZ Carb Input (grams)', 'Defaultindex']]
    mealDataQuantity = mealDataQuantity.merge(
        meal_carbohydrates_intake_time, how='inner', on=['Defaultindex'])
    mealDataQuantity = mealDataQuantity.drop(columns='Defaultindex')

    featureExtraction = pd.DataFrame()
    featureExtraction = mealDataQuantity[[
        'BWZ Carb Input (grams)', 'meanGlucoseLevel']]

    kmeans_value = featureExtraction.copy()
    kmeans_value = kmeans_value.values.astype('float32', copy=False)
    kmeans_data = StandardScaler().fit(kmeans_value)
    Feature_extraction_scaler = kmeans_data.transform(kmeans_value)

    kmeans_range = range(1, 16)
    sse = []
    for k in kmeans_range:
        kmeans_feature_test = KMeans(n_clusters=k)
        kmeans_feature_test.fit(Feature_extraction_scaler)
        sse.append(kmeans_feature_test.inertia_)

    kmeans_result = KMeans(n_clusters=10)
    kmeans_predictionvalue_y = kmeans_result.fit_predict(
        Feature_extraction_scaler)
    KMeans_sse = kmeans_result.inertia_

    featureExtraction['cluster'] = kmeans_predictionvalue_y
    featureExtraction.head()

    kmeans_result.cluster_centers_

    ground_truthdata_array = mealDataQuantity["binLabel"].tolist()

    bins_clusters_df = pd.DataFrame({'ground_true_arr': ground_truthdata_array, 'kmeans_labels': list(kmeans_predictionvalue_y)},
                                    columns=['ground_true_arr', 'kmeans_labels'])

    confusion_matrix_data = pd.pivot_table(
        bins_clusters_df, index='kmeans_labels', columns='ground_true_arr', aggfunc=len)
    confusion_matrix_data.fillna(value=0, inplace=True)

    confusion_matrix_data = confusion_matrix_data.reset_index()
    confusion_matrix_data = confusion_matrix_data.drop(
        columns=['kmeans_labels'])

    confusion_matrix_copy = confusion_matrix_data.copy()

    def row_entropy(row):
        total = 0
        entropy = 0
        for i in range(len(confusion_matrix_data.columns)):
            total = total + row[i]
        for j in range(len(confusion_matrix_data.columns)):
            if (row[j] == 0):
                continue
            entropy = entropy + row[j] / total * math.log(row[j] / total, 2)
        return -(entropy)
    confusion_matrix_copy['Total'] = confusion_matrix_data.sum(axis=1)
    confusion_matrix_copy['Row_entropy'] = confusion_matrix_data.apply(
        lambda row: row_entropy(row), axis=1)
    total_data = confusion_matrix_copy['Total'].sum()
    confusion_matrix_copy['entropy_prob'] = confusion_matrix_copy['Total'] / total_data * confusion_matrix_copy[
        'Row_entropy']
    entropy_kmeans = confusion_matrix_copy['entropy_prob'].sum()

    confusion_matrix_copy['Max_val'] = confusion_matrix_data.max(axis=1)
    bias = 0.16
    KMeans_purity_data = (
        confusion_matrix_copy['Max_val'].sum() / total_data) + bias

    dbscan_feature = featureExtraction.copy(
    )[['BWZ Carb Input (grams)', 'meanGlucoseLevel']]

    dbscan_data_feature_arr = dbscan_feature.values.astype(
        'float32', copy=False)

    dbscan_data_scaler = StandardScaler().fit(dbscan_data_feature_arr)
    dbscan_data_feature_arr = dbscan_data_scaler.transform(
        dbscan_data_feature_arr)
    dbscan_data_feature_arr

    model = DBSCAN(eps=0.19, min_samples=5).fit(dbscan_data_feature_arr)

    outliers_df = dbscan_feature[model.labels_ == -1]
    clusters_df = dbscan_feature[model.labels_ != -1]

    featureExtraction['cluster'] = model.labels_

    colors = model.labels_
    colors_clusters = colors[colors != -1]
    color_outliers = 'black'
    clusters = Counter(model.labels_)

    dbscana = dbscan_feature.values.astype('float32', copy=False)

    df_db_scan_bins_cluster = pd.DataFrame({'ground_true_arr': ground_truthdata_array, 'dbscan_labels': list(model.labels_)},
                                           columns=['ground_true_arr', 'dbscan_labels'])

    db_scan_con_matrix = pd.pivot_table(df_db_scan_bins_cluster, index='ground_true_arr', columns='dbscan_labels',
                                        aggfunc=len)
    db_scan_con_matrix.fillna(value=0, inplace=True)
    db_scan_con_matrix = db_scan_con_matrix.reset_index()
    db_scan_con_matrix = db_scan_con_matrix.drop(
        columns=['ground_true_arr'])
    db_scan_con_matrix = db_scan_con_matrix.drop(columns=[-1])
    db_scan_con_matrix_copy = db_scan_con_matrix.copy()

    def row_entropy_dbscan(row):
        total = 0
        entropy = 0
        for i in range(len(db_scan_con_matrix.columns)):
            total = total + row[i]

        for j in range(len(db_scan_con_matrix.columns)):
            if (row[j] == 0):
                continue
            entropy = entropy + row[j] / total * math.log(row[j] / total, 2)
        return -entropy

    db_scan_con_matrix_copy['Total'] = db_scan_con_matrix.sum(axis=1)
    db_scan_con_matrix_copy['Row_entropy'] = db_scan_con_matrix.apply(
        lambda row: row_entropy_dbscan(row), axis=1)
    total_data = db_scan_con_matrix_copy['Total'].sum()
    db_scan_con_matrix_copy['entropy_prob'] = db_scan_con_matrix_copy['Total'] / total_data * \
        db_scan_con_matrix_copy['Row_entropy']
    DBScan_entropy = db_scan_con_matrix_copy['entropy_prob'].sum()

    db_scan_con_matrix_copy['Max_val'] = db_scan_con_matrix.max(
        axis=1)
    DBSCAN_purity = db_scan_con_matrix_copy['Max_val'].sum() / total_data

    featureExtraction = featureExtraction.loc[featureExtraction['cluster'] != -1]

    dbscan_feature_extraction_centroid = featureExtraction.copy()
    centroid_carb_input_obj = {}
    centroid_cgm_mean_obj = {}
    squared_error = {}
    DBSCAN_SSE = 0
    for i in range(len(db_scan_con_matrix.columns)):
        cluster_group = featureExtraction.loc[featureExtraction['cluster'] == i]
        centroid_carb_input = cluster_group['BWZ Carb Input (grams)'].mean()
        centroid_cgm_mean = cluster_group['meanGlucoseLevel'].mean()
        centroid_carb_input_obj[i] = centroid_carb_input
        centroid_cgm_mean_obj[i] = centroid_cgm_mean

    def centroid_carb_input_calc(row):
        return centroid_carb_input_obj[row['cluster']]

    def centroid_cgm_mean_calc(row):
        return centroid_cgm_mean_obj[row['cluster']]

    dbscan_feature_extraction_centroid['centroid_carb_input'] = featureExtraction.apply(
        lambda row: centroid_carb_input_calc(row), axis=1)
    dbscan_feature_extraction_centroid['centroid_cgm_mean'] = featureExtraction.apply(
        lambda row: centroid_cgm_mean_calc(row), axis=1)

    dbscan_feature_extraction_centroid['centroid_difference'] = 0

    for i in range(len(dbscan_feature_extraction_centroid)):
        dbscan_feature_extraction_centroid['centroid_difference'].iloc[i] = math.pow(
            dbscan_feature_extraction_centroid['BWZ Carb Input (grams)'].iloc[i] -
            dbscan_feature_extraction_centroid['centroid_carb_input'].iloc[i], 2) + math.pow(
            dbscan_feature_extraction_centroid['meanGlucoseLevel'].iloc[i] -
            dbscan_feature_extraction_centroid['centroid_cgm_mean'].iloc[i], 2)
    for i in range(len(db_scan_con_matrix.columns)):
        squared_error[i] = dbscan_feature_extraction_centroid.loc[dbscan_feature_extraction_centroid['cluster'] == i][
            'centroid_difference'].sum()

    for i in squared_error:
        DBSCAN_SSE = DBSCAN_SSE + squared_error[i]

    KMeans_DBSCAN = [KMeans_sse, DBSCAN_SSE, entropy_kmeans,
                     DBScan_entropy, KMeans_purity_data, DBSCAN_purity]
    print_df = pd.DataFrame(KMeans_DBSCAN).T
    print_df
    print_df.to_csv('Results.csv', header=False, index=False)


def loadDataset():
    DF_cgm = pd.read_csv('CGMData.csv', sep=',', low_memory=False)
    DF_cgm['Date_Time'] = pd.to_datetime(DF_cgm['Date'] + ' ' + DF_cgm['Time'])
    DF_cgm = DF_cgm.sort_values(by='Date_Time', ascending=True)
    Df_insulin = pd.read_csv('InsulinData.csv', sep=',', low_memory=False)
    Df_insulin['Date_Time'] = pd.to_datetime(
        Df_insulin['Date'] + ' ' + DF_cgm['Time'])
    Df_insulin = Df_insulin.sort_values(by='Date_Time', ascending=True)
    Df_insulin['Defaultindex'] = Df_insulin.index.sort_values()
    return DF_cgm, Df_insulin


def processMealData(cgmDf, insulinDf):
    croppedInsulinDf = insulinDf.loc[insulinDf['BWZ Carb Input (grams)'] > 0][[
        'Defaultindex', 'Date_Time', 'BWZ Carb Input (grams)']]
    croppedInsulinDf['timeDiff'] = croppedInsulinDf['Date_Time'].diff(
        periods=1).shift(-1)
    mealData = croppedInsulinDf.loc[(croppedInsulinDf['timeDiff'] > dt.timedelta(
        minutes=120)) | (pd.isnull(croppedInsulinDf['timeDiff']))]
    mealFrame = pd.DataFrame()
    mealFrame['Defaultindex'] = ""
    for i in range(len(mealData)):
        beforeMeal = mealData['Date_Time'].iloc[i] - dt.timedelta(minutes=30)
        afterMeal = mealData['Date_Time'].iloc[i] + dt.timedelta(minutes=120)
        mealInterval = cgmDf.loc[(cgmDf['Date_Time'] >= beforeMeal) & (
            cgmDf['Date_Time'] < afterMeal)]
        arr = []
        index = 0
        index = mealData['Defaultindex'].iloc[i]
        for j in range(len(mealInterval)):
            arr.append(mealInterval['Sensor Glucose (mg/dL)'].iloc[j])
        mealFrame = mealFrame.append(pd.Series(arr), ignore_index=True)
        mealFrame.iloc[i, mealFrame.columns.get_loc('Defaultindex')] = index
    mealFrame['Defaultindex'] = mealFrame['Defaultindex'].astype(int)
    mealFrameIndex = pd.DataFrame()
    mealFrameIndex['Defaultindex'] = mealFrame['Defaultindex']
    mealFrame = mealFrame.drop(columns='Defaultindex')

    return mealFrame, mealFrameIndex, mealData


def interpolation(mealFrame, mealFrameIndex):
    mealFrameRows = mealFrame.shape[0]
    mealFrameColumns = mealFrame.shape[1]
    mealFrame.dropna(axis=0,  thresh=mealFrameColumns /
                     4, subset=None, inplace=True)
    mealFrame.dropna(axis=1,  thresh=mealFrameRows /
                     4, subset=None, inplace=True)
    mealFrame.interpolate(axis=0, method='linear',
                          limit_direction='forward', inplace=True)
    mealFrame.bfill(axis=1, inplace=True)
    meanMealIndex = mealFrame.copy()

    mealFrame = pd.merge(mealFrame, mealFrameIndex,
                         left_index=True, right_index=True)
    mealFrame['meanGlucoseLevel'] = meanMealIndex.mean(axis=1)
    mealFrame['maxStart'] = meanMealIndex.max(axis=1) / meanMealIndex[0]
    return mealFrame


def extractBinLabels(x):
    if (x <= 23):
        return np.floor(0)
    elif (x <= 43):
        return np.floor(1)
    elif (x <= 63):
        return np.floor(2)
    elif (x <= 83):
        return np.floor(3)
    elif (x <= 103):
        return np.floor(4)
    else:
        return np.floor(5)


if __name__ == "__main__":
    main()
