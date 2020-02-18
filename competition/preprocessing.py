import numpy as np
import pandas as pd
import re


class DataPreprocessor:
    """ Class used for loading and preprocessing of the races dataset """

    def process_forecast_string(self, forecast):
        """
        Extracts the weather conditions and their probabilities from a forecast string.

        Parameters:
        forecast -- forecast string in special format
        Returns:
        expanded -- dict with extracted forecast information
        weather_conditions -- weather conditions in certain order used for later mapping
        """
        weather_conditions = re.findall(r'"(.*?)"', forecast)
        probabilities = [int(x) for x in re.findall(r'i:([0-9]+)', forecast)]
        expanded = {}
        if len(weather_conditions) == len(probabilities):
            expanded = {"forecast_"+weather_conditions[i]: probabilities[i] for i in range(len(weather_conditions))}
            expanded["forecast_array"] = probabilities
        return expanded, weather_conditions

    def get_expanded_forecast_df(self, df):
        """
        Extracts the weather conditions and their probabilities from the forecast column.

        Parameters:
        df -- the races dataframe
        Returns:
        fdf -- dataframe with one column for each weather condition and one for the full array.
        weather_conditions -- weather conditions in certain order used for later mapping
        """
        forecasts = []
        for t in df.itertuples():
            expanded, weather_conditions = self.process_forecast_string(t.forecast)
            forecasts.append(expanded)
        fdf = pd.DataFrame(forecasts)
        return fdf, weather_conditions

    def clean_fuel_consumption(self, x):
        """
        Tries to cast the input as float, if it fails, returns nan

        Parameters:
        x -- the fuel consumption value
        Returns:
        value -- the input value as float
        """
        try:
            value = float(x)
        except Exception:
            value = np.nan
        return value

    def process_data(self, url, sep=";"):
        """
        Performs data loading, cleaning, transformations and feature extraction

        Parameters:
        url -- the url to the races csv file
        Returns:
        cdf -- the races dataframe
        weather_conditions -- weather conditions in certain order used for later mapping
        """
        df = pd.read_csv(url, sep=sep).dropna().reset_index(drop=True)
        df["race_driven"] = df.race_driven.apply(pd.to_datetime)
        df["fuel_consumption"] = df.fuel_consumption.apply(self.clean_fuel_consumption)
        fdf, weather_conditions = self.get_expanded_forecast_df(df)
        cdf = pd.concat([df, fdf], axis=1)
        cdf["weather_id"] = cdf.weather.apply(weather_conditions.index)
        cdf["month"] = cdf["race_driven"].dt.month
        cdf["dayofweek"] = cdf["race_driven"].dt.dayofweek
        cdf["hour"] = cdf["race_driven"].dt.hour
        cdf["forecast_argmax"] = cdf.forecast_array.apply(np.argmax)
        cdf["forecast_weather"] = cdf.forecast_argmax.apply(lambda x: weather_conditions[x])
        cdf["forecast_correct"] = (cdf.weather_id == cdf.forecast_argmax)
        cdf["global_accuracy"] = cdf.forecast_correct.sum() / cdf.shape[0]
        rolling_periods = [1000, 10000]
        for p in rolling_periods:
            cdf["rolling_%i_accuracy"%p] = cdf.forecast_correct.rolling(p).sum() / p
        cdf["player_tuple"] = cdf[["opponent", "challenger"]].values.tolist()
        cdf["player_tuple"] = cdf["player_tuple"].apply(sorted)
        cdf["player_tuple"] = cdf["player_tuple"].apply(tuple)
        return cdf, weather_conditions
