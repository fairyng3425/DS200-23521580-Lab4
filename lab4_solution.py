from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import os
import sys


# ============================================================
# CẤU HÌNH
# ============================================================
BASE = sys.argv[1] if len(sys.argv) >= 2 else "file:///C:/Users/nguye/Downloads/Lab4/data"
OUTPUT_SPARK = sys.argv[2] if len(sys.argv) >= 3 else "file:///C:/Users/nguye/Downloads/Lab4/spark_output"
OUTPUT_LOCAL = sys.argv[3] if len(sys.argv) >= 4 else "./output"


# ============================================================
# SPARK SESSION
# ============================================================
def make_spark_session():
    spark = SparkSession.builder \
        .appName("DS200_Lab4_DataFrame") \
        .master("local[*]") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")
    return spark


# ============================================================
# LOAD DATA
# ============================================================
def clean_col_names(df):
    for c in df.columns:
        new_name = c.replace("\ufeff", "").strip()
        df = df.withColumnRenamed(c, new_name)
    return df


def read_csv(spark, filename):
    path = f"{BASE}/{filename}"
    df = spark.read \
        .option("header", "true") \
        .option("sep", ";") \
        .option("inferSchema", "true") \
        .option("encoding", "UTF-8") \
        .csv(path)

    return clean_col_names(df)


def load_data(spark):
    orders = read_csv(spark, "Orders.csv")
    customers = read_csv(spark, "Customer_List.csv")
    order_items = read_csv(spark, "Order_Items.csv")
    products = read_csv(spark, "Products.csv")
    reviews = read_csv(spark, "Order_Reviews.csv")

    return orders, customers, order_items, products, reviews


def prepare_data(orders, customers, order_items, products, reviews):
    orders = orders \
        .withColumn("Order_Purchase_Timestamp", F.to_timestamp("Order_Purchase_Timestamp", "yyyy-MM-dd HH:mm")) \
        .withColumn("Order_Approved_At", F.to_timestamp("Order_Approved_At", "yyyy-MM-dd HH:mm")) \
        .withColumn("Order_Delivered_Carrier_Date", F.to_timestamp("Order_Delivered_Carrier_Date", "yyyy-MM-dd HH:mm")) \
        .withColumn("Order_Delivered_Customer_Date", F.to_timestamp("Order_Delivered_Customer_Date", "yyyy-MM-dd HH:mm")) \
        .withColumn("Order_Estimated_Delivery_Date", F.to_timestamp("Order_Estimated_Delivery_Date", "yyyy-MM-dd HH:mm"))

    customers = customers \
        .withColumn("Subscribe_Date", F.to_date("Subscribe_Date", "yyyy-MM-dd")) \
        .withColumn("First_Order_Date", F.to_date("First_Order_Date", "yyyy-MM-dd"))

    order_items = order_items \
        .withColumn("Shipping_Limit_Date", F.to_timestamp("Shipping_Limit_Date", "yyyy-MM-dd HH:mm")) \
        .withColumn("Price", F.col("Price").cast("double")) \
        .withColumn("Freight_Value", F.col("Freight_Value").cast("double"))

    products = products \
        .withColumn("Product_Weight_Gr", F.col("Product_Weight_Gr").cast("double")) \
        .withColumn("Product_Length_Cm", F.col("Product_Length_Cm").cast("double")) \
        .withColumn("Product_Height_Cm", F.col("Product_Height_Cm").cast("double")) \
        .withColumn("Product_Width_Cm", F.col("Product_Width_Cm").cast("double"))

    reviews = reviews \
        .withColumn("Review_Score", F.col("Review_Score").cast("int")) \
        .withColumn("Review_Creation_Date", F.to_timestamp("Review_Creation_Date", "yyyy-MM-dd HH:mm")) \
        .withColumn("Review_Answer_Timestamp", F.to_timestamp("Review_Answer_Timestamp", "yyyy-MM-dd HH:mm"))

    return orders, customers, order_items, products, reviews


# ============================================================
# OUTPUT HELPERS
# ============================================================
def remove_spark_output_if_exists(spark, path):
    try:
        hadoop_conf = spark.sparkContext._jsc.hadoopConfiguration()
        fs = spark.sparkContext._jvm.org.apache.hadoop.fs.FileSystem.get(hadoop_conf)
        spark_path = spark.sparkContext._jvm.org.apache.hadoop.fs.Path(path)
        if fs.exists(spark_path):
            fs.delete(spark_path, True)
    except Exception as e:
        print(f"[WARN] Cannot remove old output: {path}")
        print(f"[WARN] Reason: {e}")


def save_text_local(task_name, lines):
    os.makedirs(OUTPUT_LOCAL, exist_ok=True)
    local_file = os.path.join(OUTPUT_LOCAL, f"{task_name}.txt")

    with open(local_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[OK] Local txt file: {local_file}")


def save_df_spark(df, task_name):
    out_path = f"{OUTPUT_SPARK}/{task_name}"
    remove_spark_output_if_exists(df.sparkSession, out_path)

    df.coalesce(1) \
        .write \
        .mode("overwrite") \
        .option("header", "true") \
        .csv(out_path)

    print(f"[OK] Spark CSV output: {out_path}")


def short_text(value, width):
    value = "" if value is None else str(value)
    if len(value) <= width:
        return value
    return value[:width - 3] + "..."


def df_to_table_lines(df, title, limit=30):
    rows = df.limit(limit).collect()
    columns = df.columns

    lines = []
    lines.append("")
    lines.append("=" * 110)
    lines.append(f"{title:^110}")
    lines.append("=" * 110)

    if not rows:
        lines.append("No data.")
        return lines

    widths = []
    for c in columns:
        max_len = len(c)
        for r in rows:
            max_len = max(max_len, len(str(r[c])) if r[c] is not None else 0)

        c_lower = c.lower()

        if "id" in c_lower:
            max_len = min(max(max_len, 14), 26)
        elif "category" in c_lower or "country" in c_lower:
            max_len = min(max(max_len, 18), 32)
        elif "revenue" in c_lower or "price" in c_lower or "freight" in c_lower:
            max_len = min(max(max_len, 14), 22)
        else:
            max_len = min(max(max_len, 10), 22)

        widths.append(max_len)

    header = "  ".join(
        short_text(columns[i], widths[i]).ljust(widths[i])
        for i in range(len(columns))
    )
    lines.append(header)
    lines.append("-" * len(header))

    for r in rows:
        line_parts = []
        for i, c in enumerate(columns):
            value = short_text(r[c], widths[i])
            if isinstance(r[c], (int, float)):
                line_parts.append(value.rjust(widths[i]))
            else:
                line_parts.append(value.ljust(widths[i]))
        lines.append("  ".join(line_parts))

    lines.append("-" * len(header))
    lines.append(f"Rows shown in this table: {len(rows)}")
    return lines


def print_and_save_df(df, task_name, title, limit=50):
    total_rows = df.count()

    # Ban rut gon de in ra man hinh, giu title goc cho dep va de chup anh
    display_lines = df_to_table_lines(df, title, limit=limit)

    if total_rows > limit:
        display_lines.append("")
        display_lines.append(f"Note: Only first {limit} rows are displayed on screen.")
        display_lines.append(f"Full result has {total_rows} rows and is saved in the txt file.")

    # Ban day du de luu file txt, cung giu title goc
    full_lines = df_to_table_lines(df, title, limit=total_rows)

    print("\n".join(display_lines))
    save_text_local(task_name, full_lines)
    save_df_spark(df, task_name)
    print("=" * 110)


# ============================================================
# BÀI 1
# ============================================================
def task1(spark, orders, customers, order_items, products, reviews):
    lines = []
    lines.append("")
    lines.append("=" * 110)
    lines.append(f"{'BAI 1 - READ CSV FILES WITH INFER SCHEMA':^110}")
    lines.append("=" * 110)

    datasets = [
        ("Orders", orders),
        ("Customer_List", customers),
        ("Order_Items", order_items),
        ("Products", products),
        ("Order_Reviews", reviews)
    ]

    for name, df in datasets:
        lines.append("")
        lines.append(f"[{name}]")
        lines.append(f"Rows    : {df.count()}")
        lines.append(f"Columns : {len(df.columns)}")
        lines.append("Schema  :")
        lines.append(df._jdf.schema().treeString())

    print("\n".join(lines))
    save_text_local("bai1_schema", lines)

    schema_df = spark.createDataFrame(
        [(name, df.count(), len(df.columns), ", ".join(df.columns)) for name, df in datasets],
        ["Dataset", "Rows", "Columns", "Column_List"]
    )
    save_df_spark(schema_df, "bai1_schema")


# ============================================================
# BÀI 2
# ============================================================
def task2(spark, orders, customers, order_items):
    total_orders = orders.select("Order_ID").distinct().count()
    total_customers = customers.select("Customer_Trx_ID").distinct().count()
    total_sellers = order_items.select("Seller_ID").distinct().count()

    result = spark.createDataFrame(
        [
            ("Total Orders", total_orders),
            ("Total Customers", total_customers),
            ("Total Sellers", total_sellers)
        ],
        ["Metric", "Value"]
    )

    print_and_save_df(
        result,
        "bai2_overview_count",
        "BAI 2 - TOTAL ORDERS, CUSTOMERS AND SELLERS",
        limit=50
    )


# ============================================================
# BÀI 3
# ============================================================
def task3(orders, customers):
    result = orders.join(customers, on="Customer_Trx_ID", how="inner") \
        .groupBy("Customer_Country", "Customer_Country_Code") \
        .agg(F.countDistinct("Order_ID").alias("Total_Orders")) \
        .orderBy(F.desc("Total_Orders"), F.asc("Customer_Country"))

    print_and_save_df(
        result,
        "bai3_orders_by_country",
        "BAI 3 - ORDERS BY COUNTRY",
        limit=50
    )


# ============================================================
# BÀI 4
# ============================================================
def task4(orders):
    result = orders \
        .withColumn("Year", F.year("Order_Purchase_Timestamp")) \
        .withColumn("Month", F.month("Order_Purchase_Timestamp")) \
        .filter(F.col("Year").isNotNull() & F.col("Month").isNotNull()) \
        .groupBy("Year", "Month") \
        .agg(F.countDistinct("Order_ID").alias("Total_Orders")) \
        .orderBy(F.asc("Year"), F.desc("Month"))

    print_and_save_df(
        result,
        "bai4_orders_by_year_month",
        "BAI 4 - ORDERS BY YEAR AND MONTH",
        limit=80
    )


# ============================================================
# BÀI 5
# ============================================================
def task5(spark, reviews):
    valid_reviews = reviews \
        .filter(F.col("Review_Score").isNotNull()) \
        .filter((F.col("Review_Score") >= 1) & (F.col("Review_Score") <= 5))

    result = valid_reviews \
        .groupBy("Review_Score") \
        .agg(
            F.count("*").alias("Total_Reviews"),
            F.round(F.avg("Review_Score"), 4).alias("Avg_Score")
        ) \
        .orderBy("Review_Score")

    overall = valid_reviews.agg(
        F.count("*").alias("Valid_Reviews"),
        F.round(F.avg("Review_Score"), 4).alias("Overall_Avg_Score")
    )

    invalid_count = reviews.count() - valid_reviews.count()

    lines = df_to_table_lines(result, "BAI 5 - REVIEW SCORE DISTRIBUTION", limit=10)
    lines.append("")
    lines.append("Review score cleaning:")
    lines.append("- Keep only non-null Review_Score values.")
    lines.append("- Keep only scores from 1 to 5.")
    lines.append(f"- Invalid or null reviews removed: {invalid_count}")
    lines.append("")
    lines.extend(df_to_table_lines(overall, "BAI 5 - OVERALL REVIEW SUMMARY", limit=5))

    print("\n".join(lines))
    save_text_local("bai5_review_score", lines)
    save_df_spark(result, "bai5_review_score")


# ============================================================
# BÀI 6 - OPTIONAL
# ============================================================
def task6(orders, order_items, products):
    order_2024 = orders \
        .withColumn("Year", F.year("Order_Purchase_Timestamp")) \
        .filter(F.col("Year") == 2024) \
        .select("Order_ID")

    items_2024 = order_2024.join(order_items, on="Order_ID", how="inner") \
        .withColumn("Revenue", F.col("Price") + F.col("Freight_Value"))

    result = items_2024.join(products, on="Product_ID", how="left") \
        .withColumn("Product_Category_Name", F.coalesce(F.col("Product_Category_Name"), F.lit("Unknown"))) \
        .groupBy("Product_Category_Name") \
        .agg(
            F.round(F.sum("Revenue"), 2).alias("Total_Revenue_2024"),
            F.count("*").alias("Total_Items_Sold"),
            F.countDistinct("Order_ID").alias("Total_Orders")
        ) \
        .orderBy(F.desc("Total_Revenue_2024"))

    print_and_save_df(
        result,
        "bai6_revenue_2024_by_category",
        "BAI 6 - REVENUE IN 2024 BY PRODUCT CATEGORY",
        limit=50
    )


# ============================================================
# BÀI 7 - OPTIONAL
# ============================================================
def task7(order_items, products, reviews):
    valid_reviews = reviews \
        .filter(F.col("Review_Score").isNotNull()) \
        .filter((F.col("Review_Score") >= 1) & (F.col("Review_Score") <= 5)) \
        .select("Order_ID", "Review_Score")

    product_sales = order_items \
        .groupBy("Product_ID") \
        .agg(
            F.count("*").alias("Quantity_Sold"),
            F.round(F.sum(F.col("Price") + F.col("Freight_Value")), 2).alias("Total_Revenue")
        )

    product_review = order_items.join(valid_reviews, on="Order_ID", how="left") \
        .groupBy("Product_ID") \
        .agg(
            F.round(F.avg("Review_Score"), 4).alias("Avg_Review_Score"),
            F.count("Review_Score").alias("Review_Count")
        )

    result = product_sales.join(product_review, on="Product_ID", how="left") \
        .join(products.select("Product_ID", "Product_Category_Name"), on="Product_ID", how="left") \
        .withColumn("Product_Category_Name", F.coalesce(F.col("Product_Category_Name"), F.lit("Unknown"))) \
        .select(
            "Product_ID",
            "Product_Category_Name",
            "Quantity_Sold",
            "Total_Revenue",
            "Avg_Review_Score",
            "Review_Count"
        ) \
        .orderBy(F.desc("Quantity_Sold"), F.desc("Total_Revenue"))

    top_product = result.limit(1)
    total_rows = result.count()

    display_lines = []
    display_lines.extend(df_to_table_lines(
        top_product,
        "BAI 7 - PRODUCT WITH HIGHEST QUANTITY SOLD",
        limit=1
    ))

    display_lines.append("")
    display_lines.append("Giai thich:")
    display_lines.append("- Quantity_Sold duoc tinh bang cach dem so dong san pham trong Order_Items theo tung Product_ID.")
    display_lines.append("- San pham o bang tren la san pham co so luong ban ra cao nhat.")
    display_lines.append("- Neu nhieu san pham co cung Quantity_Sold, Total_Revenue duoc dung lam tieu chi sap xep tiep theo.")

    display_lines.extend(df_to_table_lines(
        result,
        "BAI 7 - ALL PRODUCTS",
        limit=50
    ))

    if total_rows > 50:
        display_lines.append("")
        display_lines.append("Note: Only first 50 product rows are displayed on screen.")
        display_lines.append(f"Full result has {total_rows} rows and is saved in the txt file.")

    full_lines = []
    full_lines.extend(df_to_table_lines(
        top_product,
        "BAI 7 - PRODUCT WITH HIGHEST QUANTITY SOLD",
        limit=1
    ))

    full_lines.append("")
    full_lines.append("Giai thich:")
    full_lines.append("- Quantity_Sold duoc tinh bang cach dem so dong san pham trong Order_Items theo tung Product_ID.")
    full_lines.append("- San pham o bang tren la san pham co so luong ban ra cao nhat.")
    full_lines.append("- Neu nhieu san pham co cung Quantity_Sold, Total_Revenue duoc dung lam tieu chi sap xep tiep theo.")

    full_lines.extend(df_to_table_lines(
        result,
        "BAI 7 - ALL PRODUCTS",
        limit=total_rows
    ))

    print("\n".join(display_lines))
    save_text_local("bai7_top_products_and_avg_review", full_lines)
    save_df_spark(result, "bai7_top_products_and_avg_review")
    print("=" * 110)


# ============================================================
# BÀI 10 - OPTIONAL
# ============================================================
def task10(order_items):
    seller_base = order_items \
        .withColumn("Revenue", F.col("Price") + F.col("Freight_Value")) \
        .groupBy("Seller_ID") \
        .agg(
            F.round(F.sum("Revenue"), 2).alias("Total_Revenue"),
            F.countDistinct("Order_ID").alias("Total_Orders"),
            F.count("*").alias("Total_Items_Sold"),
            F.round(F.avg("Revenue"), 2).alias("Avg_Revenue_Per_Item")
        )

    window_spec = Window.orderBy(F.desc("Total_Revenue"), F.desc("Total_Orders"))

    result = seller_base \
        .withColumn("Rank", F.dense_rank().over(window_spec)) \
        .select(
            "Rank",
            "Seller_ID",
            "Total_Revenue",
            "Total_Orders",
            "Total_Items_Sold",
            "Avg_Revenue_Per_Item"
        ) \
        .orderBy("Rank")

    print_and_save_df(
        result,
        "bai10_seller_ranking",
        "BAI 10 - SELLER RANKING BY REVENUE AND ORDERS",
        limit=50
    )


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    spark = make_spark_session()

    try:
        orders, customers, order_items, products, reviews = load_data(spark)

        orders, customers, order_items, products, reviews = prepare_data(
            orders, customers, order_items, products, reviews
        )

        task1(spark, orders, customers, order_items, products, reviews)
        task2(spark, orders, customers, order_items)
        task3(orders, customers)
        task4(orders)
        task5(spark, reviews)

        # Chon 3 bai tu bai 6 den bai 10
        task6(orders, order_items, products)
        task7(order_items, products, reviews)
        task10(order_items)

    finally:
        spark.stop()
