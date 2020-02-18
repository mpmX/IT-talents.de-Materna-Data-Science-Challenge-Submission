import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class RaceActivity:
    """ Class containing methods for number of races analysis. """

    def plot_time_aggregated_n_races(self, df, freq="1M", figsize=None):
        """
        Aggregates all races by freq and plots the number of races for each bin.

        Parameters:
        df -- the races dataframe
        freq -- aggregation frequency
        figsize -- size of the figure
        """
        df = df.copy()
        df.index = df.race_driven
        fig, ax = plt.subplots(2, sharex=True, figsize=figsize)
        fig.suptitle("Number of races per %s"%freq, fontsize=16)
        for i, scale in enumerate(["linear", "log"]):
            agg = df.groupby(pd.Grouper(freq=freq, label="right")).id.count()
            agg.index.name = "Date"
            agg.plot(ax=ax[i])
            ax[i].set_yscale(scale)
            ax[i].set_ylabel("# races")
            ax[i].title.set_text("%s scale" % scale)
        plt.show()

    def plot_aggregated_by_features(self, df, features=["hour", "dayofweek", "month"], figsize=None):
        """
        Aggregates all races by each input feature column and plots the number of races for each bin.

        Parameters:
        df -- the races dataframe
        features -- the aggregation columns
        figsize -- size of the figure
        """
        fig, ax = plt.subplots(len(features), figsize=figsize)
        fig.suptitle("number of races binned by date features", fontsize=16)
        for i, f in enumerate(features):
            df.groupby(f).id.count().plot.bar(ax=ax[i])
        plt.show()
