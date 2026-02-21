# simple code to convert csv to txt, helped by chatgpt

import csv

def csv_column_to_txt(csv_path, txt_path, column_name):
    with open(csv_path, newline='', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        
        with open(txt_path, 'w', encoding='utf-8') as txt_file:
            for row in reader:
                value = row[column_name]
                txt_file.write(value + "\n")

csv_column_to_txt('data/test.csv', 'data/test.txt', 'context')