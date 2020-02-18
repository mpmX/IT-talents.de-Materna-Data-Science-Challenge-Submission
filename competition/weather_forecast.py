import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report


class WeatherForecastAnalysis:
    """ Class containing methods to analyze the weather forecast. """

    def __init__(self, weather_conditions):
        self.weather_conditions = weather_conditions
        self.subfigure_ids = "abcdefg"


    def plot_forecast_boxplots_by_trackid(self, df):
        """
        Plots the forecast array boxplots for each track id.

        Parameters:
        df -- the races dataframe
        """
        n_tracks = df.track_id.unique().shape[0]
        forecast_cols = ["forecast_"+x for x in self.weather_conditions]
        fig, ax = plt.subplots(2, len(forecast_cols), sharey=True)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.suptitle("Forecast Distribution", fontsize=16)
        for i, fc in enumerate(forecast_cols):
            sns.boxplot(y=fc, data=df, palette="Set3", ax=ax[0][i])
            sns.boxplot(x="track_id", y=fc, data=df, palette="Set3", ax=ax[1][i])
        plt.show()

    def plot_weather_distribution_by_trackid(self, df):
        """
        Plots actual weather distribution for each track id.

        Parameters:
        df -- the races dataframe
        """
        n_columns = int(df.track_id.unique().shape[0]/2)
        fig, ax = plt.subplots(2, n_columns)
        fig.suptitle("Weather by track", fontsize=16)
        wdf = pd.DataFrame(index=df.weather.unique())
        for i, (gn, g) in enumerate(df.sort_values("track_id").groupby("track_id")):
            weather_dist = g.groupby("weather").weather.count()/g.shape[0]
            weather_dist = wdf.copy().join(weather_dist)
            weather_dist.plot.bar(ax=ax[(i-1) // n_columns, i%n_columns])
            ax[(i-1) // n_columns, i%n_columns].title.set_text(gn)
        plt.show()



    def plot_conditional_accuracy_analysis(self, df):
        """
        Plots forecast accuracy over time and conditional accuracies.

        Parameters:
        df -- the races dataframe
        """
        condition_columns = ["track_id", "month", "dayofweek", "forecast_weather", "weather"]
        fig = plt.figure()
        gs = fig.add_gridspec(2,len(condition_columns))
        fig.suptitle("Forecast Accuracy", fontsize=16)
        ax = fig.add_subplot(gs[0, :])
        ax.title.set_text("over time")
        df[["global_accuracy"]+[c for c in df.columns if c.startswith("rolling")]].plot(ax=ax)
        for i, condition in enumerate(condition_columns):
            results = []
            for gn, g in df.groupby(condition):
                accuracy = g.forecast_correct.sum() / g.shape[0]
                results.append({condition: gn, "accuracy": accuracy})
            rdf = pd.DataFrame(results)
            ax = fig.add_subplot(gs[1, i])
            rdf.plot.bar(x=condition, y="accuracy", ax=ax)
            ax.set_ylim(0, 1)
            ax.title.set_text("(%s) by %s" % (self.subfigure_ids[i], condition))
        plt.show()

    def plot_forecast_confusion_matrix(self, df):
        """
        Plots the forecast confusion matrix.

        Parameters:
        df -- the races dataframe
        """
        normalizations = ["all", "true"]
        titles = ["normalized over all samples\n(global frequency score)", "normalized over true label\n(true label local frequency score)"]
        fig, ax = plt.subplots(1,2)
        fig.suptitle("Forecast Confusion Matrix", fontsize=16)
        for i, normalization in enumerate(normalizations):
            cm = confusion_matrix(y_true=df.weather.values, y_pred=df.forecast_weather.values, sample_weight=None,
                                  labels=self.weather_conditions, normalize=normalization)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.weather_conditions)
            disp.plot(include_values=True, cmap="OrRd", ax=ax[i], xticks_rotation='horizontal')
            ax[i].title.set_text(titles[i])
        plt.show()


    def plot_precision_recall(self, df):
        """
        Plots precision and recall of the weather forecast.

        Parameters:
        df -- the races dataframe
        """
        cr = classification_report(df.weather.values, df.forecast_weather, output_dict=True)
        cr.pop("accuracy")
        infos = []
        for i in cr.keys():
            info = cr[i]
            for k,v in info.items():
                infos.append({"forecast": i, "value": v, "metric": k})
        fdf = pd.DataFrame(infos)
        fig, ax = plt.subplots(1)
        ax.title.set_text("Precision / Recall / F1-Score")
        sns.barplot(y="forecast", x="value", hue="metric", data=fdf[fdf.metric!="support"], orient="h", ax=ax)
        plt.show()
