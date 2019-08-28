# -*- coding: utf-8 -*-
"""Helper utilities and decorators."""
import sys
import time
import urllib.request
import zipfile
import yaml
from pathlib import Path

from tqdm import tqdm
import nltk
from colored import fg

import kiwi
from kiwi import constants as const
from IPython.display import Markdown, display
nltk.download('punkt')

# Download and extract methods heavily based on TorchNLP
def download_kiwi(url, directory="trained_models"):
    """Download file at `url`. Extract if needed"""

    def _check_if_downloading(file):
        print("Checking if download in progress", file=sys.stderr)
        size = file.stat().st_size
        time.sleep(1)
        return size != file.stat().st_size

    directory = Path(directory)
    if not directory.exists():
        directory.mkdir(parents=True)
    elif not directory.is_dir():
        raise ValueError(f'Not a directory: {directory}')

    # get filename from url
    print("Getting filename", file=sys.stderr)
    filename = url.split("/")[-1]

    filepath = Path(directory) / Path(filename)

    target_directory = filepath.parent / filepath.stem
    # if the file isn't there or doesn't have the correct size, download it
    print("Checking if file already downloaded", file=sys.stderr)
    download_file = False

    if filepath.exists() or filepath.with_suffix('').exists():
        pass
    else:
        download_file = True

    if download_file:
        print("Downloading", file=sys.stderr)
        # urllib.request.urlretrieve(url, filename=filepath)
        with tqdm(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
            urllib.request.urlretrieve(url, filename=filepath, reporthook=_reporthook(t))
        print("Download has finished.", file=sys.stderr)

    print("Extracting {}".format(filepath), file=sys.stderr)
    return _maybe_extract(filepath, target_directory=target_directory)


def _maybe_extract(compressed_path, target_directory=None):
    """checks if files have already been extracted and extracts them if not"""

    extension = compressed_path.suffix

    if target_directory is None:
        target_directory = compressed_path.parent / compressed_path.stem

    if not target_directory.exists():
        if "zip" in extension:
            with zipfile.ZipFile(compressed_path, "r") as zipped:
                zipped.extractall(target_directory)
        else:
            print("File type not supported", file=sys.stderr)

    print("Done extracting", file=sys.stderr)

    return target_directory


def _reporthook(t):
    """ ``reporthook`` to use with ``urllib.request`` that prints the process of the download.
    """
    last_b = [0]

    def inner(b=1, bsize=1, tsize=None):
        """
        Args:
            b (int, optional): Number of blocks just transferred [default: 1].
            bsize (int, optional): Size of each block (in tqdm units) [default: 1].
            tsize (int, optional): Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return inner

def save_config(yaml_config, name):
    """ Writes yaml config to file"""
    with open(name, 'w') as outfile:
        yaml.dump(yaml_config, outfile, default_flow_style=False)

# Utils sourced fromn Kiwi_viz demo. Available at github.com/unbabel/OpenKiwi/demo/kiwi_viz.ipynb

def printmd(text):
    display(Markdown(text))
    
def make_color(text, color):
    start = "<span style='color:" + color + "'>"
    return start + text + "</span>"


def get_color(bad_prob, threshold):
    return 'green' if bad_prob < threshold else 'red'

def highlight(text_bad_prob):
    text, bad_prob, threshold = text_bad_prob
    if bad_prob >= threshold:
        text = '*' + text + '*'
    return make_color(text, get_color(bad_prob, threshold))


def KiwiViz(model, source, mt, threshold=0.5):
    source = ' '.join(nltk.word_tokenize(source))
    mt = ' '.join(nltk.word_tokenize(mt))
    model_out = model.predict({const.SOURCE: [source.lower()], const.TARGET: [mt.lower()]})
    bad_probs = model_out[const.TARGET_TAGS][0]
    gap_probs = model_out[const.GAP_TAGS][0]
    highlight_words = list(map(highlight, zip(mt.split(), bad_probs, [threshold for b in bad_probs])))
    visualization = ''
    BAD_GAP = make_color('_', 'red')
    for gap_prob, word in zip(gap_probs[:-1], highlight_words):
        visualization += ' ' if gap_prob < threshold else BAD_GAP + ' '
        visualization += word
    visualization += '' if gap_probs[-1] < threshold else ' ' + BAD_GAP
    print('HTER: {}'.format(model_out[const.SENTENCE_SCORES][0]))

    printmd(visualization)

