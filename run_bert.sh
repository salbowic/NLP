#!/bin/bash

python3 main_bert.py --batch_size 16 --review_type_name imdb --max_length 256 --version 3 --dr 0.0

python3 main_bert.py --batch_size 16 --review_type_name imdb --max_length 256 --version 3 --dr 0.1

python3 main_bert.py --batch_size 16 --review_type_name imdb --max_length 256 --version 3 --dr 0.2

python3 main_bert.py --batch_size 16 --review_type_name imdb --max_length 256 --version 3 --dr 0.3

python3 main_bert.py --batch_size 16 --review_type_name imdb --max_length 256 --version 3 --dr 0.4


python3 main_bert.py --batch_size 16 --review_type_name mcdonald --max_length 256 --version 3 --dr 0.0

python3 main_bert.py --batch_size 16 --review_type_name mcdonald --max_length 256 --version 3 --dr 0.1

python3 main_bert.py --batch_size 16 --review_type_name mcdonald --max_length 256 --version 3 --dr 0.2

python3 main_bert.py --batch_size 16 --review_type_name mcdonald --max_length 256 --version 3 --dr 0.3

python3 main_bert.py --batch_size 16 --review_type_name mcdonald --max_length 256 --version 3 --dr 0.4


python3 main_bert.py --batch_size 16 --review_type_name twitter --max_length 128 --version 3 --dr 0.0

python3 main_bert.py --batch_size 16 --review_type_name twitter --max_length 128 --version 3 --dr 0.1

python3 main_bert.py --batch_size 16 --review_type_name twitter --max_length 128 --version 3 --dr 0.2

python3 main_bert.py --batch_size 16 --review_type_name twitter --max_length 128 --version 3 --dr 0.3

python3 main_bert.py --batch_size 16 --review_type_name twitter --max_length 128 --version 3 --dr 0.4