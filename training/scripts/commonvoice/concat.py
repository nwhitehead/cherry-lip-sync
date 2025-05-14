"""

Concat multiple parquet files together.

Files should be result of running extract.py script.

"""

import argparse
import pandas
import pyarrow.parquet as pq

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', nargs='+')
    parser.add_argument('--output', type=str, default='out.parquet')
    args = parser.parse_args()
    schema = pq.ParquetFile(args.input[0]).schema_arrow
    with pq.ParquetWriter(args.output, schema=schema) as writer:
        for file in args.input:
            writer.write_table(pq.read_table(file, schema=schema))
            print(file)