from helpers.GetEnv import GetEnv
import json
import pandas as pd
from CustomFactories.SparkSessionFactory import SparkSessionFactory
from .module_helpers.GetConnection import GetConnection

def generate_collection_text(collection_row):
        return f"""
                Match: {collection_row['home_team']} vs {collection_row['away_team']}
                Date: {collection_row['date']}
                Competition: {collection_row['tournament']}
                Result: {collection_row['home_team']} {collection_row['home_score']} - {collection_row['away_score']} {collection_row['away_team']}
                Venue City: {collection_row['city']}
                Venue Country: {collection_row['country']}
                Is Neutral: {'Yes' if collection_row['is_neutral'] == 1 else 'No'}
                Winner: {'Draw' if collection_row['home_score'] == collection_row['away_score'] 
                        else collection_row['home_team'] if collection_row['home_score'] > collection_row['away_score'] 
                        else collection_row['away_team']}
                """.strip()

if __name__ == "__main__":

    _env = GetEnv.get_env_variables()

    # Get and init the connection for collection and vector embedding model
    _con = GetConnection()
    client, embedding_model, collection = _con.initConnection()
    

    sparkSession = SparkSessionFactory.create_spark_session()
    featured_delta_path = f"{_env['DATA_LAKE_PATH']}/featured_result/result_data"

    dataframe = sparkSession.read.format('delta').load(featured_delta_path)
    sparkSession.stop()
    
    pdf = dataframe.toPandas()
    BATCH_SIZE = int(_env['VECTOR_EMBEDDING_BATCH_SIZE'])

    for i in range(0, len(pdf), BATCH_SIZE):
        iteration = pdf.iloc[i:i+BATCH_SIZE]

        texts = []
        metadatas = []
        ids = []

        for index, row_data in iteration.iterrows():
            text = generate_collection_text(row_data)
            texts.append(text)
            ids.extend([
                f"Match_{row_data['inc_id']}_match_result"
            ])

            metadatas.append({
                    "text":      text,
                    "home_team": row_data['home_team'],
                    "away_team": row_data['away_team'],
                    "winner":    'Draw' if row_data['home_score'] == row_data['away_score'] else row_data['home_team'] if row_data['home_score'] > row_data['away_score'] else row_data['away_team'],
                    "date":      row_data['date'],
                    "type":      "match_result"
            })
            
        vectors = embedding_model.encode( # Converts our vector into dimensions as per the embeddin model we use (dimensions = numbers) 
            texts,
            batch_size=64,           # smaller for CPU
            convert_to_numpy=True,
            show_progress_bar=True
        )

        collection.upsert(
            ids=ids,
            embeddings=vectors.tolist(),
            metadatas=metadatas,
            documents=texts
        )
        print(f"Loaded {min(i+BATCH_SIZE, len(pdf))}/{len(pdf)}")

    print("Encoding done ✅")

    # collection.upsert(
    #     ids=["id1", "id2", "id3"],
    #     documents=[
    #         "This is a document about pineapple",
    #         "This is a document about madurai",
    #         "This is a document about oranges"
    #     ]
    # )

    # results = collection.query(
    #     query_texts=["This is a query document about madurai"], # Chroma will embed this for you
    #     n_results=2 # how many results to return
    # )
    # print(json.dumps(results, indent=2))
    