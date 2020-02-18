import numpy as np
import pandas as pd
import random

class RacePrediction:
    """ Contains a simple race prediction method. """

    def predict_winner(self, df, player_id1, player_id2):
        """
        Predicts the winner of a race between player_id1 and player_id2.
        Uses historical results of races between the two players.

        Parameters:
        df -- the races dataframe
        player_id1 -- player id 1
        player_id2 -- player id 2
        Returns:
        -1 if no data exists, otherwise the winner player id.
        """
        player_tuple = tuple(sorted([player_id1, player_id2]))
        historical_races = df[df.player_tuple == player_tuple]
        if historical_races.shape[0] > 0:
            win_counts = historical_races.groupby("winner").winner.count()
            if win_counts.shape[0] == 1:
                return win_counts.index.values[0]
            else:
                if win_counts.loc[player_id1] == win_counts.loc[player_id2]:
                    return random.choice([player_id1, player_id2])
                elif win_counts.loc[player_id1] > win_counts.loc[player_id2]:
                    return player_id1
                else:
                    return player_id2
        else:
            # print("no race history between the two players! maybe try %s" % str(df.sample(1).player_tuple.iloc[0]))
            return -1


    def evaluate_predictions(self, train_df, test_df):
        """
        Evaluates the simple prediction method on a test set.

        Parameters:
        train_df -- training part of the races dataframe
        test_df -- testing part of the races dataframe
        Returns:
        dict containing the accuracy on the test set and the percentage of races where we had data
        """
        test_df["winner_prediction"] = test_df.player_tuple.apply(lambda x: self.predict_winner(train_df, x[0], x[1]))
        predicted_races = test_df[test_df.winner_prediction > 0]
        accuracy = (predicted_races.winner == predicted_races.winner_prediction).sum() / predicted_races.shape[0]
        return {"accuracy": accuracy, "races_predicted": predicted_races.shape[0] / test_df.shape[0]}
