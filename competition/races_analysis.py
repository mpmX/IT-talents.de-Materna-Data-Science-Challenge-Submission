import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report

from preprocessing import *
from race_activity import *
from player_analysis import *
from pair_analysis import *
from weather_forecast import *
from race_prediction import *

""" Experimentation script """

if __name__ == "__main__":
    dpp = DataPreprocessor()
    df, weather_conditions = dpp.process_data("data/races.csv")

    ra = RaceActivity()
    ra.plot_time_aggregated_n_races(df)
    ra.plot_aggregated_by_date_features(df)

    pa = PlayerAnalysis()
    pdf, track_ids = pa.get_players_df(df)
    pa.plot_flux(pdf)
    pa.plot_player_profile(df, pdf, 1)>

    X = np.stack(pdf.track_probabilities.values)
    pa.plot_pca(X, c=pdf.track_diversity.values, title="Track preference PCA")
    cluster_prediction = pa.plot_kmeans(X, 3)
    pdf["track_preference_cluster"] = cluster_prediction
    pa.plot_cluster_means(pdf, "track_preference_cluster", track_ids)

    pa = PairAnalysis()
    pair_counts, ppdf = pa.calculate_pair_statistics(df)
    pa.plot_pair_counts(pair_counts)
    pa.plot_pair_profile(df, (46,48))


    fa = WeatherForecastAnalysis(weather_conditions)
    fa.plot_weather_by_trackid(df)
    fa.plot_forecast_boxplots_by_trackid(df)
    fa.plot_conditional_accuracy_analysis(df)
    fa.plot_forecast_confusion_matrix(df)
    fa.plot_precision_recall(df)


    rp = RacePrediction()
    rp.predict_race(df, 48, 46)
    test_split = 0.1
    test_split_idx = int(df.shape[0]*test_split)
    train_df = df.iloc[:-test_split_idx]
    test_df = df.iloc[-test_split_idx:].copy()
    pres = rp.evaluate_predictions(train_df, test_df)
