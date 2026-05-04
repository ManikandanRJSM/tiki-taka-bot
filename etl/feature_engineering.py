import requests
import pandas as pd
from io import StringIO
from CustomFactories.SparkSessionFactory import SparkSessionFactory
from pyspark.sql.functions import col, isnan, to_date, when, count, monotonically_increasing_id, lit, sum, desc
from pyspark.sql import functions as F
from .app_constants.constants import result_map, K_map
from helpers.GetEnv import GetEnv
import json
import argparse
import numpy as np



if __name__ == '__main__':

    _env = GetEnv.get_env_variables()

    sparkSession = SparkSessionFactory.create_spark_session()
    featured_delta_path = f"{_env['DATA_LAKE_PATH']}/featured_result/result_data"
    featured_csv_path = f"{_env['DATA_LAKE_PATH']}/featured_result/result_data_csv"
    delta_path = f"{_env['DATA_LAKE_PATH']}/preprocessed_result"

    dataframe = sparkSession.read.format('delta').load(delta_path)
    
    get_total_wins = dataframe.filter(col('match_result').isin([1, 2])).count()
    get_total_goals = dataframe.agg(sum('total_goals')).alias('sum_goals').collect()
    get_total_goals_home = dataframe.agg(sum('home_score')).alias('sum_home_goals').collect()
    get_total_goals_away = dataframe.agg(sum('away_score')).alias('sum_away_goals').collect()
    total_matches = dataframe.count()

    # Calculate the default values if the teams dont have past 5 meetings
    global_average = round(get_total_wins/total_matches, 2)
    global_goals_socred_avg = round(get_total_goals[0][0]/total_matches, 2)


    dataframe.createOrReplaceTempView('PreprocessTable')

    team_form = sparkSession.sql(f"""
                        CREATE OR REPLACE TEMP VIEW teamHistory AS
                        -- home team row
                        SELECT
                            formated_date,
                            home_team AS team,
                            if(match_result = {result_map['home_win']}, 1, 0) AS win,
                            home_score goals,
                            away_score goal_conced
                        FROM preprocessTable

                        UNION ALL

                        -- away team row
                        SELECT
                            formated_date,
                            away_team AS team,
                            if(match_result = {result_map['away_win']}, 1, 0) AS win,
                            away_score goals,
                            home_score goal_conced
                        FROM preprocessTable     
                    """)
    

    sparkSession.sql(f"""
                        CREATE OR REPLACE TEMP VIEW teamForm AS
                        SELECT formated_date, 
                        team, 
                        coalesce(round(avg(win) over(partition by team order by formated_date rows between 5 preceding and 1 preceding), 2), {global_average}) AS win_rate_5,
                        coalesce(round(avg(goals) over(partition by team order by formated_date rows between 5 preceding and 1 preceding), 2), {global_goals_socred_avg}) AS avg_goalsrate_5,
                        coalesce(round(avg(goal_conced) over(partition by team order by formated_date rows between 5 preceding and 1 preceding), 2), ( SELECT ROUND(AVG(goal_conced), 2) AS default_conced FROM teamHistory )) AS avg_goals_conced_last_5
                        FROM teamHistory
                     
    """)
    
    featured_result = sparkSession.sql(f"""
        SELECT
            pt.*,
            home_tf.win_rate_5 AS home_team_win_rate_5,
            away_tf.win_rate_5 AS away_team_win_rate_5,
                                       
            home_tf.avg_goalsrate_5 AS home_team_avg_goals_rate_5,
            away_tf.avg_goalsrate_5 AS away_team_avg_goals_rate_5,
                                       
            home_tf.avg_goals_conced_last_5 AS home_avg_goals_conceded_last5,
            away_tf.avg_goals_conced_last_5 AS away_avg_goals_conceded_last5,
                                       
            coalesce(round(AVG(
                if(
                    (pt.home_team = Q.home_team AND Q.match_result = {result_map['home_win']}) OR
                    (pt.home_team = Q.away_team AND Q.match_result = {result_map['away_win']}),
                    1, 0
                )
            ), 2), 0.00) AS h2h_win_ratio_home,
            coalesce(round(AVG(
                Q.home_score + Q.away_score
            ), 2), 0.00) AS h2h_avg_goals

        FROM preprocessTable pt

        LEFT JOIN teamForm home_tf
            ON pt.formated_date = home_tf.formated_date
            AND pt.home_team = home_tf.team

        LEFT JOIN teamForm away_tf
            ON pt.formated_date = away_tf.formated_date
            AND pt.away_team = away_tf.team

        LEFT JOIN preprocessTable Q
            ON (
                (Q.home_team = pt.home_team AND Q.away_team = pt.away_team) OR
                (Q.home_team = pt.away_team  AND Q.away_team = pt.home_team)
            )
            AND Q.formated_date < pt.formated_date

        GROUP BY
            pt.date,
            pt.formated_date,
            pt.total_goals,
            pt.is_neutral,
            pt.match_importance,
            pt.home_team,
            pt.away_team,
            pt.home_score,
            pt.away_score,
            pt.tournament,
            pt.city,
            pt.country,
            pt.neutral,
            pt.match_result,
            home_tf.win_rate_5,
            away_tf.win_rate_5,
            home_tf.avg_goalsrate_5,
            away_tf.avg_goalsrate_5,
            home_tf.avg_goals_conced_last_5,
            away_tf.avg_goals_conced_last_5

        ORDER BY pt.formated_date
    """)

    
    # featured_result.show(2)
    featured_result = featured_result.withColumn('inc_id', monotonically_increasing_id() + 1).withColumns({
        'home_elo' : lit(None).cast("double"),
        'away_elo' : lit(None).cast("double"),
        'last_5_home_win_rate_home_only' : lit(None).cast("double"),
        'last_5_away_win_rate_away_only' : lit(None).cast("double")
    })
    
    pdf = featured_result.toPandas() 
    pdf = pdf.sort_values(['formated_date', 'inc_id']).reset_index(drop=True)

    # Pre-compute all-time default rates (used when a team has < 6 prior home/away appearances)
    home_total      = pdf.groupby('home_team').size()
    home_wins_total = pdf[pdf['match_result'] == result_map['home_win']].groupby('home_team').size().reindex(home_total.index, fill_value=0)
    away_total      = pdf.groupby('away_team').size()
    away_wins_total = pdf[pdf['match_result'] == result_map['away_win']].groupby('away_team').size().reindex(away_total.index, fill_value=0)

    home_default_rate = (home_wins_total / home_total).round(2).to_dict()
    away_default_rate = (away_wins_total / away_total).round(2).to_dict()

    # Sequential loop - ELO is stateful; win-rate histories built incrementally row by row
    elo_dict     = {}
    home_history = {}   # team -> list of home match wins (1/0) in chronological order
    away_history = {}   # team -> list of away match wins (1/0) in chronological order
    default_elo  = 1500
    n            = len(pdf)

    home_elo_arr  = np.empty(n, dtype=float)
    away_elo_arr  = np.empty(n, dtype=float)
    elo_diff_arr  = np.empty(n, dtype=float)
    exp_prob_arr  = np.empty(n, dtype=float)
    home_rate_arr = np.empty(n, dtype=float)
    away_rate_arr = np.empty(n, dtype=float)

    for i, row in enumerate(pdf.itertuples(index=False)):
        home_team  = row.home_team
        away_team  = row.away_team
        result     = row.match_result
        tournament = row.tournament

        # last 5 home win rate — O(1) slice of pre-built history list
        h_hist = home_history.get(home_team, [])
        if len(h_hist) < 6:
            home_rate_arr[i] = home_default_rate.get(home_team, 0.0)
        else:
            last5 = h_hist[-5:]
            home_rate_arr[i] = round(float(np.mean(last5)), 2)

        # last 5 away win rate — O(1) slice of pre-built history list
        a_hist = away_history.get(away_team, [])
        if len(a_hist) < 6:
            away_rate_arr[i] = away_default_rate.get(away_team, 0.0)
        else:
            last5 = a_hist[-5:]
            away_rate_arr[i] = round(float(np.mean(last5)), 2)

        # ELO — step 1: fetch
        home_elo = elo_dict.get(home_team, default_elo)
        away_elo = elo_dict.get(away_team, default_elo)
        E_home   = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))

        home_elo_arr[i] = home_elo
        away_elo_arr[i] = away_elo
        elo_diff_arr[i] = home_elo - away_elo
        exp_prob_arr[i] = E_home

        # ELO — step 2: actual result
        if result == result_map['home_win']:
            S_home = 1
        elif result == result_map['draw']:
            S_home = 0.5
        else:
            S_home = 0

        # ELO — step 3: update after match
        K = K_map.get(tournament, 20)
        elo_dict[home_team] = round(float(home_elo + K * (S_home - E_home)), 2)
        elo_dict[away_team] = round(float(away_elo + K * ((1 - S_home) - (1 - E_home))), 2)

        # Append to histories AFTER computing features so current match is excluded
        home_history.setdefault(home_team, []).append(1 if result == result_map['home_win'] else 0)
        away_history.setdefault(away_team, []).append(1 if result == result_map['away_win'] else 0)

    # Bulk column assignment — avoids 45K × 6 individual pdf.at[] writes
    pdf['home_elo']                       = home_elo_arr
    pdf['away_elo']                       = away_elo_arr
    pdf['elo_diff']                       = elo_diff_arr.round(2)
    pdf['expected_prob']                  = exp_prob_arr.round(2)
    pdf['last_5_home_win_rate_home_only'] = home_rate_arr
    pdf['last_5_away_win_rate_away_only'] = away_rate_arr

    
            

    feature_df = sparkSession.createDataFrame(pdf)

    feature_df = feature_df.withColumns({
        'home_score' : col('home_score').cast("int"),
        'away_score' : col('away_score').cast("int"),
        'last_5_home_win_rate_home_only' : col('last_5_home_win_rate_home_only').cast('double'),
        'last_5_away_win_rate_away_only' : col('last_5_away_win_rate_away_only').cast('double')
    })

    feature_df.write.format('csv').mode("overwrite").option("header", True).save(featured_csv_path)
    feature_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(featured_delta_path) # option("mergeSchema", "true") for schema evolution

    # if DeltaTable.isDeltaTable(spark_session, delta_path):
    #     deltaTable = DeltaTable.forPath(spark_session, delta_path)

    #     deltaTable.alias("target").merge(
    #     data_df.alias("source"),
    #     "target.date = source.date AND target.home_team = source.home_team AND target.away_team = source.away_team AND target.match_result = source.match_result"
    #     ).whenMatchedUpdate(condition=condition_to_check, set={ col: f"source.{col}" for col in pre_process_schema }) \
    #     .whenNotMatchedInsertAll() \
    #     .execute()
    # else:
    #     data_df.write.format("delta").mode("overwrite").save(delta_path) # it saves delta log
        
    with open(f"{_env['DATA_LAKE_PATH']}/featured_result/elo/elo.json", 'w') as f:
        json.dump(elo_dict, f, indent=2)

    print("Feature Engineering Done............!")
    sparkSession.stop()