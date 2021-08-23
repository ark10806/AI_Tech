import csv

path = '/opt/ml/input/new_test/info.csv'
path = './test.csv'

a = ['a'] * 10
b = [x for x in range(10)]

with open(path, 'w', newline='\n') as csvfile:
    wr = csv.writer(csvfile, delimiter=',')
    for i in range(10):
        wr.writerow([a[i], b[i]])

def write_csv(path: str, fname: str, pred: int):
    with open(path, 'w', newline='\n') as csvfile:
    wr = csv.writer(csvfile, delimiter=',')
    for i in range(10):
        wr.writerow([a[i], b[i]])