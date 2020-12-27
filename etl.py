import configparser
import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, to_timestamp, monotonically_increasing_id
import pandas as pd
from pyspark.sql.types import *

config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']=config['AWS']['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY']=config['AWS']['AWS_SECRET_ACCESS_KEY']

from_date = datetime.datetime(2020,1,1,0,0,0)
to_date = datetime.datetime(2020,6,30,23,59,59)

def create_spark_session():
    """
      The create_spark_session function creates a new spark session or retrieves the previous spark session if already created.
    """
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark


def process_zone_data(spark, output_data):
    #Input File
    taxi_zone_file = "taxi_zones.json"

    #Creating a PySpark DF
    zone_spark_df = spark.read.option("multiline", "true").json(taxi_zone_file)

    # Selecting columns
    zone_table = zone_spark_df.select(["LocationID", "Borough", "Zone"])

    # Writing the Zone Dimension Table to S3 Bucket in parquet format
    zone_table.write.parquet(output_data + "/zone_table/", mode="overwrite")
    zone_table.show(10)


def process_payment_data(spark, output_data):
    # Creating payment Panda dataframe
    payment_data = {'PaymentID': [1, 2, 3, 4, 5, 6],
                    'Payment_type': ['Credit card', 'Cash', 'No charge', 'Dispute', 'Unknown', 'Voided trip']}
    df_payment = pd.DataFrame(payment_data)

    # Creating a Rate Spark Dataframe from the existing Panda DF
    paymentSchema = StructType([StructField("PaymentID", IntegerType(), True), StructField("Payment_type", StringType(), True)])
    payment_spark_df = spark.createDataFrame(df_payment, schema=paymentSchema)

    # Writing the Payment Dimension Table to S3 Bucket in parquet format
    payment_spark_df.write.parquet(output_data + "/payment_table/", mode="overwrite")


def process_rate_data(spark, output_data):
    # Creating Rate Panda dataframe
    rate_data = {'RatecodeID': [1, 2, 3, 4, 5, 6],
                 'Rate_type': ['Standard rate', 'JFK', 'Newark', 'Nassau or Westchester', 'Negotiated fare', 'Group ride']}

    df_rate = pd.DataFrame(rate_data)

    # Creating a Rate Spark Dataframe from the existing Panda DF
    rateSchema = StructType([StructField("RatecodeID", IntegerType(), True), StructField("Rate_type", StringType(), True)])
    rate_spark_df = spark.createDataFrame(df_rate, schema=rateSchema)

    # Writing the Rate Dimension Table to S3 Bucket in parquet format
    rate_spark_df.write.parquet(output_data + "/rate_table/", mode="overwrite")


def process_weather_data(spark, output_data):
    # Input file
    weather_file = "weather.csv"

    # Creating a PySpark DF
    weather_spark_df = spark.read.option("multiline","true").option("header", "true").csv(weather_file)

    # Removing duplicate data in the weather DF. (keeping only JFK, removing others).
    weather_spark_df = weather_spark_df.filter(weather_spark_df['name'] == "JFK INTERNATIONAL AIRPORT, NY US")

    # Convert DATE and DropTime to TIMESTAMP.
    format = udf(lambda x: x.replace("T", " ") + " 2020")
    # add column
    weather_spark_df = weather_spark_df.withColumn("TIME", format(weather_spark_df["DATE"]))
    weather_spark_df = weather_spark_df.withColumn("ts", to_timestamp(weather_spark_df["TIME"], "MM-dd HH:mm:ss yyyy"))
    # delete old column
    weather_spark_df = weather_spark_df.drop("DATE")
    weather_spark_df = weather_spark_df.drop("TIME")

    # Filter Weather data so that it matches the time period of taxi data.
    weather_spark_df = weather_spark_df.filter(weather_spark_df["ts"] >= from_date).filter(weather_spark_df["ts"] <= to_date)

    # Renaming Columns of weather_spark_df
    weather_spark_df = weather_spark_df.withColumnRenamed("HLY-CLOD-PCTBKN", "Clouds_Broken_Percentage") \
        .withColumnRenamed("HLY-DEWP-NORMAL", "Dew_Point") \
        .withColumnRenamed("HLY-HTDH-NORMAL", "Heating_Degree_Hours") \
        .withColumnRenamed("HLY-HIDX-NORMAL", "Heat_Index_Mean") \
        .withColumnRenamed("HLY-WIND-AVGSPD", "Wind_Speed") \
        .withColumnRenamed("HLY-WIND-PCTCLM", "Percentage_Calm") \
        .withColumnRenamed("HLY-CLOD-PCTOVC", "Clouds_Overcast_Percentage") \
        .withColumnRenamed("HLY-PRES-NORMAL", "Sea_Level_Pressure") \
        .withColumnRenamed("HLY-TEMP-NORMAL", "Temperature") \
        .withColumnRenamed("HLY-WCHL-NORMAL", "Wind_Chill") \
        .withColumnRenamed("HLY-WIND-1STDIR", "Wind_Direction")

    weather_spark_df.createOrReplaceTempView("weather_view")
    # udf to get month, day, hour
    spark.udf.register("getMonth", lambda x: int(x.month))
    spark.udf.register("getDay", lambda x: int(x.day))
    spark.udf.register("getHour", lambda x: int(x.hour))
    spark.udf.register("getTS", lambda x: int(x.timestamp))

    # Query for Weather Dimension Table
    weather_table = spark.sql("""
        SELECT
            Clouds_Broken_Percentage,
            Dew_Point,
            Heating_Degree_Hours,
            Heat_Index_Mean,
            Wind_Speed,
            Percentage_Calm,
            Clouds_Overcast_Percentage,
            Sea_Level_Pressure,
            Temperature,
            Wind_Chill,
            Wind_Direction,
            getMonth(ts) AS Month,
            getDay(ts) AS Day,
            getHour(ts) AS Hour
        FROM weather_view;
    """)

    # Writing the Weather Dimension Table to S3 Bucket in parquet format
    weather_table.write.partitionBy("month", "day").parquet(output_data + "/zone_table/", mode="overwrite")


def process_taxi_data(spark, output_data):
    # Input file
    taxi_file_url = "https://data.cityofnewyork.us/api/views/pkmi-4kfn/rows.csv"

    # Creating a Panda DF
    df_taxi = pd.read_csv(taxi_file_url)

    # Creating a new struct for taxi data so that data from the existing Panda DF can be
    # copied instead of downloading it again
    taxiSchema = StructType([StructField("VendorID", FloatType(), True)
                                , StructField("lpep_pickup_datetime", StringType(), True)
                                , StructField("lpep_dropoff_datetime", StringType(), True)
                                , StructField("store_and_fwd_flag", StringType(), True)
                                , StructField("RatecodeID", FloatType(), True)
                                , StructField("PULocationID", IntegerType(), True)
                                , StructField("DOLocationID", IntegerType(), True)
                                , StructField("passenger_count", FloatType(), True)
                                , StructField("trip_distance", FloatType(), True)
                                , StructField("fare_amount", FloatType(), True)
                                , StructField("extra", FloatType(), True)
                                , StructField("mta_tax", FloatType(), True)
                                , StructField("tip_amount", FloatType(), True)
                                , StructField("tolls_amount", FloatType(), True)
                                , StructField("ehail_fee", FloatType(), True)
                                , StructField("improvement_surcharge", FloatType(), True)
                                , StructField("total_amount", FloatType(), True)
                                , StructField("payment_type", FloatType(), True)
                                , StructField("trip_type", FloatType(), True)
                                , StructField("congestion_surcharge", FloatType(), True)])

    # Creating a taxi Spark Dataframe from the existing Panda DF
    taxi_spark_df = spark.createDataFrame(df_taxi, schema=taxiSchema)

    # Payment is unknown in few records of the taxi DF
    taxi_spark_df = taxi_spark_df.filter(taxi_spark_df["payment_type"] != 5)

    # Zone table has two unknown values, id 264 and 265
    taxi_spark_df = taxi_spark_df.filter(taxi_spark_df["PULocationID"] != 264).filter(taxi_spark_df["PULocationID"] != 265)

    # PickUpTime and DropTime is in String format
    # add column with new time stamp column
    taxi_spark_df = taxi_spark_df.withColumn("PickupTS", to_timestamp(taxi_spark_df["lpep_pickup_datetime"],"MM/dd/yyyy hh:mm:ss a"))
    taxi_spark_df = taxi_spark_df.withColumn("DropTS", to_timestamp(taxi_spark_df["lpep_dropoff_datetime"],"MM/dd/yyyy hh:mm:ss a"))
    # drop column with new time stamp column
    taxi_spark_df = taxi_spark_df.drop("lpep_pickup_datetime", "lpep_dropoff_datetime")

    # Range selected for the project is 1st Jan to 31st June
    taxi_spark_df = taxi_spark_df.filter(taxi_spark_df["PickupTS"] >= from_date).filter(taxi_spark_df["DropTS"] <= to_date)

    # Some data in taxi DF are NULL
    taxi_spark_df = taxi_spark_df.filter(taxi_spark_df["PickupTS"].isNotNull() == True) \
        .filter(taxi_spark_df["DropTS"].isNotNull() == True) \
        .filter(taxi_spark_df["RatecodeID"].isNotNull() == True) \
        .filter(taxi_spark_df["PULocationID"].isNotNull() == True) \
        .filter(taxi_spark_df["DOLocationID"].isNotNull() == True) \
        .filter(taxi_spark_df["passenger_count"].isNotNull() == True) \
        .filter(taxi_spark_df["trip_distance"].isNotNull() == True) \
        .filter(taxi_spark_df["fare_amount"].isNotNull() == True) \
        .filter(taxi_spark_df["payment_type"].isNotNull() == True) \
        .filter(taxi_spark_df["extra"].isNotNull() == True) \
        .filter(taxi_spark_df["mta_tax"].isNotNull() == True) \
        .filter(taxi_spark_df["tip_amount"].isNotNull() == True) \
        .filter(taxi_spark_df["tolls_amount"].isNotNull() == True) \
        .filter(taxi_spark_df["improvement_surcharge"].isNotNull() == True) \
        .filter(taxi_spark_df["total_amount"].isNotNull() == True) \
        .filter(taxi_spark_df["congestion_surcharge"].isNotNull() == True)

    # Adding an unique id column in taxi_spark_df
    taxi_spark_df = taxi_spark_df.withColumn("id", monotonically_increasing_id())
    taxi_spark_df.createOrReplaceTempView("taxi_view")

    # UDF to convert to get Timestamp
    spark.udf.register("getTS", lambda x: datetime.datetime.timestamp(x))

    # Query for Trip Dimension Table
    trip_table = spark.sql("""
        SELECT
            id AS trip_id,
            getTS(PickupTS) AS pickup_timestamp,
            getTS(DropTS) AS drop_timestamp,
            fare_amount,
            extra,
            tip_amount,
            tolls_amount,
            congestion_surcharge,
            total_amount
        FROM taxi_view""")

    # Writing the Weather Dimension Table to S3 Bucket in parquet format
    trip_table.write.parquet(output_data + "/trip_table/", mode="overwrite")

    # Query for Fact Table
    fact_table = spark.sql("""
        SELECT
            id AS trip_id,
            getMonth(PickupTS) AS Month,
            getDay(PickupTS) AS Day,
            getHour(PickupTS) AS Hour,
            RatecodeID AS rate_code,
            payment_type AS payment_code,
            PULocationID AS pick_up_loc_id,
            DOLocationID AS drop_loc_id,
            total_amount,
            passenger_count,
            trip_distance
        FROM  taxi_view""")

    # Writing the Fact Table to S3 Bucket in parquet format, partitioned by "Month","Day","Hour"
    fact_table.write.partitionBy("Month", "Day", "Hour").parquet(output_data + "/fact_table/", mode="overwrite")


def main():
    """
      The main function -
      I. Creates a spark session
     II. Parses the various file and stores it in the S3 Bucket in Parquet format
    """
    spark = create_spark_session()

    output_data = "s3a://my-udacity-course-bucket/capstone-project/"

    process_zone_data(spark,output_data)
    process_payment_data(spark,output_data)
    process_rate_data(spark,output_data)
    process_weather_data(spark,output_data)
    process_taxi_data(spark,output_data)


if __name__ == "__main__":
    main()