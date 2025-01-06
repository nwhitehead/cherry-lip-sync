import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
import argparse
import random
import torchaudio
import pyloudnorm as pyln
import glob

table = pq.read_table(os.path.join('.', 'data', f'lipsync.parquet')).to_pandas()
print(table)