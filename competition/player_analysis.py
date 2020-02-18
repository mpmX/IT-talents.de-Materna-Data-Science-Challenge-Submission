import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

class PlayerAnalysis:
    """ Contains functions used for a player centric analysis of the races data """

    def get_players_df(self, df):
        """
        Extracts players and their features from the races dataframe.
        Uses vectorized operations where possible.

        Parameters:
        df -- the races dataframe
        Returns:
        pdf -- the player dataframe
        track_ids -- list of track ids in the order that was used to generate the track probabilities array.
                     Can be used to map array indices to track ids.
        """
        players = sorted(list(set(df.challenger.unique().tolist() + df.opponent.unique().tolist())))
        pdf = pd.DataFrame({"player_id": players})
        pdf.index = pdf.player_id
        pdf["n_wins"] = df.groupby("winner").winner.count()
        pdf["n_challenger"] = df.groupby("challenger").challenger.count()
        pdf["n_opponent"] = df.groupby("opponent").opponent.count()
        pdf["money_won"] = df.groupby("winner").money.sum()
        money_lost = []
        track_diversity = []
        track_probabilities = []
        track_ids = sorted(df.track_id.unique())
        for p in pdf.itertuples():
            player_races = df[((df.challenger==p.player_id) | (df.opponent==p.player_id))]
            lost_races = player_races[player_races.winner != p.player_id]
            ml = lost_races.money.sum()
            money_lost.append(ml)
            tps = player_races.groupby("track_id").track_id.count()
            tps = tps / tps.sum()
            tpdf = pd.DataFrame(index=track_ids).join(tps).fillna(0)
            tpdf.columns = ["track_probability"]
            track_probabilities.append(tpdf["track_probability"].values.tolist())
            td = tpdf["track_probability"].std()
            track_diversity.append(td)
        pdf["track_diversity"] = track_diversity
        pdf["track_diversity"] = 1-pdf["track_diversity"]
        pdf["track_diversity"] = MinMaxScaler().fit_transform(pdf[["track_diversity"]])
        pdf["track_probabilities"] = track_probabilities
        pdf["money_lost"] = money_lost
        pdf[["n_wins", "n_challenger", "n_opponent",
             "money_won", "money_lost"]] = pdf[["n_wins", "n_challenger", "n_opponent",
                                                "money_won", "money_lost"]].fillna(0).astype(int)
        pdf["net_profit"] = pdf.money_won-pdf.money_lost
        pdf["n_races"] = pdf.n_challenger + pdf.n_opponent
        pdf["n_losses"] = pdf.n_races-pdf.n_wins
        pdf["win_ratio"] = pdf.n_wins/pdf.n_races
        pdf["race_balance"] = pdf.n_wins-pdf.n_losses
        pdf["avg_profit_per_race"] = pdf.net_profit/pdf.n_races
        pdf["first_race_challenger"] = df.groupby("challenger").race_driven.min()
        pdf["first_race_opponent"] = df.groupby("opponent").race_driven.min()
        pdf["first_race"] = pdf[["first_race_challenger", "first_race_opponent"]].min(axis=1)
        pdf["last_race_challenger"] = df.groupby("challenger").race_driven.max()
        pdf["last_race_opponent"] = df.groupby("opponent").race_driven.max()
        pdf["last_race"] = pdf[["last_race_challenger", "last_race_opponent"]].max(axis=1)
        pdf = pdf.drop(["first_race_challenger", "first_race_opponent", "last_race_challenger", "last_race_opponent"], axis=1)
        pdf["lifespan"] = pdf.last_race-pdf.first_race
        return pdf, track_ids

    def plot_flux(self, pdf, freq="1M"):
        """
        Plots the player influx/outflux and flux.

        Parameters:
        pdf -- the players dataframe
        freq -- the aggregation frequency. see pandas docs for valid values. Default = 1M
        """
        pdf = pdf.copy()
        pdf.index = pdf.first_race
        player_influx = pdf.groupby(pd.Grouper(freq=freq, label="right")).first_race.count()
        pdf.index = pdf.last_race
        player_outflux = pdf.groupby(pd.Grouper(freq=freq, label="right")).last_race.count()
        idf = pd.concat([player_influx, player_outflux], axis=1)
        idf.index.name = "Date"
        idf["flux"] = (player_influx-player_outflux)
        fig, ax = plt.subplots(2, sharex=True)
        fig.suptitle("Monthly player flux", fontsize=16)
        idf[["first_race", "last_race"]].plot(ax=ax[0])
        idf[["flux"]].plot(ax=ax[1])
        ax[1].hlines([0], idf.index.values[0], idf.index.values[-1], color="black")
        plt.show()

    def plot_player_profile(self, df, pdf, player):
        """
        Plots the player profile of a given player id.

        Parameters:
        df -- the races dataframe
        pdf -- the players dataframe
        player -- the player id
        """
        races = df[(df.challenger==player) | (df.opponent==player)].copy()
        races.index = races.race_driven
        races = races.sort_index()
        races["cum_n_races"] = 1
        races["cum_n_races"] = races["cum_n_races"].cumsum()
        aggs = ["hour", "month", "dayofweek", "track_id"]

        fig = plt.figure()
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.suptitle("Player profile\n%i"%player, fontsize=16)
        gs = fig.add_gridspec(2,len(aggs))
        for i, group in enumerate(aggs):
            ax = fig.add_subplot(gs[0, i])
            races.groupby(group).id.count().plot.bar(ax=ax)
            ax.title.set_text("race count by %s"%group)

        ax = fig.add_subplot(gs[1, :-1])
        races["cum_n_races"].plot(ax=ax)
        ax.title.set_text("Cumulative number of races")
        ax = fig.add_subplot(gs[1, -1])
        winratio = pd.DataFrame([pdf[pdf.player_id==player].win_ratio, 1-pdf[pdf.player_id==player].win_ratio], index=["win", "lose"])
        winratio.columns = ["win/lose %"]
        winratio["win/lose %"].plot.pie(ax=ax, autopct='%1.1f%%')
        ax.title.set_text("win distribution")
        plt.show()

    def plot_pca(self, X, c, title=""):
        """
        Plots the two-dimensional PCA latent space.

        Parameters:
        X -- input array with shape [n_samples, n_features]
        c -- the array used for coloring
        title -- the plot title
        """
        pred = PCA(n_components=2, random_state=7).fit_transform(X)
        plt.scatter(x=pred[:,0], y=pred[:,1], c=c, cmap="RdYlGn", alpha=0.25)
        plt.gca().title.set_text(title)
        plt.colorbar()
        plt.show()

    def plot_kmeans(self, X, k):
        """
        Performs KMean clustering on the input data and visualizes
        the two-dimensional PCA latent space color coded by class membership.

        Parameters:
        X -- input array with shape [n_samples, n_features]
        k -- number of clusters
        Returns:
        pred -- array of predicted cluster ids.
        """
        pred = KMeans(n_clusters=k, random_state=0).fit_predict(X)
        pca = PCA(n_components=2, random_state=7).fit_transform(X)
        plt.scatter(x=pca[:,0], y=pca[:,1], c=pred, cmap="tab10", alpha=0.9)
        plt.gca().title.set_text("Track preference PCA\n colored by kmeans k=%i"%k)
        plt.show()
        return pred

    def plot_cluster_means(self, pdf, cluster_column, track_ids):
        """
        Plots the mean feature vector for each cluster.

        Parameters:
        pdf -- the players dataframe
        cluster_column -- the column that hold the cluster id.
        track_ids -- track ids mapping list.
        """
        mean_feature_vectors = {}
        for c in sorted(pdf[cluster_column].unique()):
            tps = np.stack(pdf[pdf[cluster_column]==c].track_probabilities.values)
            mean_feature_vectors["cluster_%i"%c] = np.mean(tps, axis=0)
        df = pd.DataFrame(mean_feature_vectors)
        df.index = track_ids
        ax = df.plot(title="mean feature vectors for each cluster", cmap="tab10")
        ax.set_xticks(track_ids)
        ax.set_xlabel("track_id")
        plt.show()
