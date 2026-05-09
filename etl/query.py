from CustomFactories.SparkSessionFactory import SparkSessionFactory

if __name__ == "__main__":

    # query = input("Your query → ").strip()

    sparkSession = SparkSessionFactory.create_spark_session()

    query = sparkSession.sql(f"""
        SELECT inc_id, home_team, away_team, tournament, date, formated_date, match_result, home_score, away_score FROM parquet.`./data_lake/featured_result/result_data` where home_team = 'Brazil' and away_team = 'Germany' and tournament = 'FIFA World Cup' order by date desc limit 10;
    """)
    query.show()

    sparkSession.stop()
