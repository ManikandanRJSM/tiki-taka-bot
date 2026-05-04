import requests
import pandas as pd
from io import StringIO
from CustomFactories.SparkSessionFactory import SparkSessionFactory
from pyspark.sql.functions import col, isnan, to_date, when, count, monotonically_increasing_id, lit, sum, desc
from pyspark.sql import functions as F
from .app_constants.constants import result_map, K_map
from delta.tables import DeltaTable
from helpers.GetEnv import GetEnv
from GlobalConstants.constants import pre_process_schema
import json
import argparse
import numpy as np



if __name__ == '__main__':

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--mode", required=True, type=str, choices=['test', 'train'], default='train', help="Preprocess data for train / test")
    # parser.add_argument("--start_date", required=True, type=str, default='1872-11-30', help="Start date")
    # parser.add_argument("--end_date", required=True, type=str, default='2024-12-31', help="End date")

    # args = parser.parse_args()

    # mode = args.mode
    # start_date = args.start_date
    # end_date = args.end_date
    _env = GetEnv.get_env_variables()

    url = "https://raw.githubusercontent.com/ManikandanRJSM/international_results/master/results.csv"
    

    spark_session = SparkSessionFactory.create_spark_session()

    # if mode == 'test':
    #     delta_path = f"{_env['DATA_LAKE_PATH']}/pre_processed_data/test_data"
    # else:
    delta_path = f"{_env['DATA_LAKE_PATH']}/preprocessed_result"

    response = requests.get(url)
    pdf = pd.read_csv(StringIO(response.text))
    mapping_expr = F.create_map([lit(x) for kv in K_map.items() for x in kv])

    # Convert Pandas -> Spark DataFrame
    df = spark_session.createDataFrame(pdf)

    upcoming_matches_df = df.filter(isnan(col('home_score')) & isnan(col('away_score')))

    #withColumn('date', to_date("date")) convert date as string to date
    df = df.withColumn('formated_date', to_date("date")) \
        .filter(~isnan(col('home_score')) & ~isnan(col('away_score')))
    
    # Remove the duplicate entry
    cleaned_df = df.dropDuplicates()
    
    cleaned_df = cleaned_df.withColumns( {
        'home_score' : col('home_score').cast("int"),
        'away_score' : col('away_score').cast("int")
    })
    
    # Quarantine DF
    quarantine_df = df.exceptAll(cleaned_df)

    # if end_date is not None:
    #     cleaned_df = cleaned_df.filter( col('formated_date').between(start_date, end_date) )
    
    cleaned_df = cleaned_df.withColumn(
        'match_result', when(col('home_score') > col('away_score'), result_map['home_win']) \
        .when(col('home_score') == col('away_score'), result_map['draw']) \
        .when(col('home_score') < col('away_score'), result_map['away_win'])
    )

    data_df = cleaned_df.withColumns({
        'total_goals' : col('home_score') + col('away_score'),
        'is_neutral' : when(col('neutral') == True, 1).otherwise(0),
        'match_importance' : F.coalesce(mapping_expr[col('tournament')], lit(20))
    })

    condition_to_check = " OR ".join([ f"target.{i} != source.{i}" for i in pre_process_schema ])

    if DeltaTable.isDeltaTable(spark_session, delta_path):
        deltaTable = DeltaTable.forPath(spark_session, delta_path)

        deltaTable.alias("target").merge(
        data_df.alias("source"),
        "target.date = source.date AND target.home_team = source.home_team AND target.away_team = source.away_team AND target.match_result = source.match_result"
        ).whenMatchedUpdate(condition=condition_to_check, set={ col: f"source.{col}" for col in pre_process_schema }) \
        .whenNotMatchedInsertAll() \
        .execute()
    else:
        data_df.write.format("delta").mode("overwrite").save(delta_path) # it saves delta log

    print("Preprocessing Done............!")
    # feature_extraction(spark_session, data_df, _env['DATA_LAKE_PATH'])
        
    spark_session.stop()
