import os
import bz2

from six.moves import urllib


def download_lenta(path: str = 'data/raw'):
    output_dir = os.path.join(path, 'lenta')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    url = 'https://github.com/yutkin/Lenta.Ru-News-Dataset/releases/download/v1.1/lenta-ru-news.csv.bz2'
    print("downloading url ", url)

    data = urllib.request.urlopen(url)
    file_path = os.path.join(output_dir, os.path.basename(url))
    print(file_path)
    with open(file_path, 'wb') as f:
        f.write(data.read())

    print("Extracting data")
    with open(file_path, 'rb') as source, open(os.path.join(output_dir, 'lenta.csv'), 'wb') as dest:
        dest.write(bz2.decompress(source.read()))

    os.remove(file_path)

    return output_dir


if __name__ == '__main__':
    download_lenta()
