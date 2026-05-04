from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip

class SparkSessionFactory:
    @staticmethod
    def create_spark_session():

        #local[2] -> 2 cores if * use all cores in machine

        SparkSession_builder = SparkSession.builder \
                .appName("MyApp") \
                .master("local[1]") \
                .config("spark.sql.shuffle.partitions", "2") \
                .config("spark.executor.memory", "2g") \
                .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
                .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        spark = configure_spark_with_delta_pip(SparkSession_builder).getOrCreate()
        # print(spark._jvm.org.apache.hadoop.util.VersionInfo.getVersion())
        # .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \ # Enable this line if you are using Delta Lake and want to use the DeltaCatalog for managing your tables. This is necessary for features like time travel, schema evolution, and other Delta Lake functionalities.
        return spark
