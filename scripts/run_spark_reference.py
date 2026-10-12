"""Run PySpark MLlib FPGrowth on a transaction dataset; write reference output.

Dev-only validation tool — the fpm library itself never depends on PySpark.

Usage:
  python scripts/run_spark_reference.py DATASET.dat.gz OUT_DIR MIN_SUPPORT [MIN_CONFIDENCE]

Reads a FIMI-format dataset (one transaction per line, space-separated item
ids, optionally gzipped), runs Spark FPGrowth, and writes two parquet dirs:
  OUT_DIR/itemsets   (items: array<string>, freq: long)
  OUT_DIR/rules      (antecedent, consequent, confidence, lift, support)
Plus OUT_DIR/meta.json with N, minSupport, minConfidence, and the dataset
sha256. Item ids are kept as strings by Spark; the parity checker maps back.
"""

import hashlib
import json
import os
import sys


def dataset_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    dataset, out_dir, min_support = sys.argv[1], sys.argv[2], float(sys.argv[3])
    min_confidence = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0

    from pyspark.sql import SparkSession
    from pyspark.ml.fpm import FPGrowth

    spark = (
        SparkSession.builder.master("local[*]")
        .appName("fpm-spark-reference")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    opener = open
    if dataset.endswith(".gz"):
        import gzip

        opener = gzip.open
    rows = []
    with opener(dataset, "rt") as fh:
        for line in fh:
            toks = line.split()
            # Spark FPGrowth rejects duplicate items within a transaction
            # (fpm tolerates them; set membership is unaffected either way).
            rows.append((list(dict.fromkeys(toks)),))  # item ids stay strings
    n = len(rows)
    df = spark.createDataFrame(rows, ["items"])

    model = FPGrowth(minSupport=min_support, minConfidence=min_confidence,
                     itemsCol="items").fit(df)
    os.makedirs(out_dir, exist_ok=True)
    model.freqItemsets.write.mode("overwrite").parquet(os.path.join(out_dir, "itemsets"))
    model.associationRules.write.mode("overwrite").parquet(os.path.join(out_dir, "rules"))

    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump({"n_transactions": n, "min_support": min_support,
                   "min_confidence": min_confidence, "dataset": os.path.abspath(dataset),
                   "dataset_sha256": dataset_sha256(dataset)}, fh, indent=2)
    print(f"Spark reference written to {out_dir} "
          f"({model.freqItemsets.count()} itemsets, {model.associationRules.count()} rules, "
          f"N={n})")
    spark.stop()


if __name__ == "__main__":
    main()
