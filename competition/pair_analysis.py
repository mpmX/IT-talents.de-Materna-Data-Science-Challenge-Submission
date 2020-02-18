import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class PairAnalysis:
    """Contains functions for the analysis from a pair perspective."""

    def calculate_pair_statistics(self, df):
        """
        Extracts player pairs from the races dataframe, counts the number of occurence
        and calculates the number of unique matchups per player.

        Parameters:
        df -- the races dataframe
        Returns:
        pair_counts -- series with (player_id1, player_id2) tuples as index and
                       the respective counts as values.
        cdf -- dataframe with player id as index and matchup statistics as values
        """
        pair_counts = df.groupby("player_tuple").id.count()
        pair_counts.name = "pairs"

        player_ids = []
        for pair in pair_counts.index.values:
            player_ids.append(pair[0])
            player_ids.append(pair[1])

        player_unique_pair_counts = {pid: player_ids.count(pid) for pid in set(player_ids)}
        cdf = pd.DataFrame(player_unique_pair_counts.values(),
                           index=player_unique_pair_counts.keys(),
                           columns=["unique_matchups"]).sort_values("unique_matchups")
        cdf.index.name = "player_id"
        cdf["unique_matchups_all_player_pct"] = cdf.unique_matchups / len(set(player_ids))
        return pair_counts, cdf

    def plot_pair_counts(self, pair_counts):
        """
        Draws a boxplot of pair matchup counts.

        Parameters:
        pair_counts -- series of matchup counts
        """
        pair_counts.name = "pairs"
        fig, ax = plt.subplots(2)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.suptitle("Player pair matchup count", fontsize=16)
        for i, scale in enumerate(["linear", "log"]):
            pair_counts.plot.box(ax=ax[i], vert=False)
            ax[i].set_xscale(scale)
            if i > 0:
                ax[i].set_xlabel("# matchups")
            ax[i].title.set_text("%s x scale"%scale)
        plt.show()

    def plot_pair_profile(self, df, pair):
        """
        Plots a the pair profile visualization for a given pair.

        Parameters:
        df -- the races dataframe
        pair -- tuple containing the two sorted player ids
        """
        races = df[df.player_tuple==pair].copy()
        winratio = races.groupby("winner").id.count() / races.shape[0]
        races.index = races.race_driven
        races = races.sort_index()
        races["cum_n_races"] = 1
        races["cum_n_races"] = races["cum_n_races"].cumsum()
        aggs = ["hour", "month", "dayofweek", "track_id"]
        fig = plt.figure()
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.suptitle("Pair profile\n%s"%str(pair), fontsize=16)
        gs = fig.add_gridspec(2,len(aggs))
        for i, group in enumerate(aggs):
            ax = fig.add_subplot(gs[0, i])
            races.groupby(group).id.count().plot.bar(ax=ax)
            ax.title.set_text("race count by %s"%group)
        ax = fig.add_subplot(gs[1, :-1])
        races["cum_n_races"].plot(ax=ax)
        ax.title.set_text("Cumulative number of races")
        ax = fig.add_subplot(gs[1, -1])
        winratio.plot.pie(ax=ax, autopct="%1.1f%%")
        ax.title.set_text("win distribution")
        plt.show()
