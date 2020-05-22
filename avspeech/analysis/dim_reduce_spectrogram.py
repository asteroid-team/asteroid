from pathlib import Path
from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import umap
from tsne import bh_sne
from sklearn.decomposition import IncrementalPCA


def plot_tsne(generator):
    X = next(generator(return_full=True))
    X_2d = bh_sne(X.astype(np.float64))
    plt.scatter(X_2d[:, 0], X_2d[:, 1])
    plt.show()
    return X_2d

def plot_umap(generator):
    X = next(generator(return_full=True))
    X_2d = umap.UMAP().fit_transform(X)
    plt.scatter(X_2d[:, 0], X_2d[:, 1])
    plt.show()
    return X_2d

def plot_pca(generator, generator2, batch_size=50):
    i_pca = IncrementalPCA(n_components=2, batch_size=batch_size)
    generator = generator(return_full=False, batch=batch_size)
    generator2 = generator2(return_full=False, batch=batch_size)

    X_2d = []
    for x in generator2:
        i_pca.partial_fit(x)
    for x in generator:
        X_2d.append(i_pca.transform(x))
    X_2d = np.squeeze(np.vstack(X_2d))
    print(X_2d.shape)
    plt.scatter(X_2d[:, 0], X_2d[:, 1])
    plt.show()

def generate_data(files, shape, filter_files):
    # check mem use
    n_floats = np.prod(shape)
    bytes_req = n_floats * 4
    print(f"{bytes_req:,} bytes")
    high_mem = False

    if bytes_req > 2**30:
        print(f"High mem use: {bytes_req:,} bytes")
        high_mem = True

    def _yield(return_full=True, batch=500):
        if not high_mem and return_full:
            X = np.empty(shape, dtype=np.float64)
        else:
            X = np.empty((batch, *shape[1:]), dtype=np.float32)

        X = []
        for i, f in tqdm(enumerate(files), total=len(files)):
            if filter_files == "mixed" and not f.stem.isnumeric():
                continue
            elif filter_files == "raw" and f.stem.isnumeric():
                continue
            x = np.load(f).astype(np.float32)
            X.append(x)
            length = len(X)

            if length != 0 and length % batch == 0 and not return_full:
                X = np.reshape(np.array(X), (length, -1))
                yield X
                X = []
        if return_full:
            length = len(X)
            X = np.reshape(np.array(X), (length, -1))
            yield X

    return _yield

def main(args):
    files = list(args.input_dir.rglob("*npy"))[:args.total_files]

    # get dim
    shape = np.load(files[0]).shape
    total_files = len(files)

    X_shape = (total_files, *shape)

    generator = generate_data(files, X_shape, args.filter_files)
    method = args.method

    if method == "tsne":
        plot_tsne(generator)
    elif method == "pca":
        generator2 = generate_data(files, X_shape, args.filter_files)
        plot_pca(generator, generator2)
    elif method == "umap":
        plot_umap(generator)

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--input-dir", type=Path, default=Path("../../data/train/spec"), help="Input directory where all the numpy array are stored.")
    parser.add_argument("--method", '-m', type=str, default="tsne", help="Dimensionality Reduction method", choices=["tsne", "pca", "umap"])
    parser.add_argument("--total-files", '-n', type=int, default=100, help="Total file to use")
    parser.add_argument("--filter-files", type=str, default="none", help="Select only particular kind of file", choices=["none", "mixed", "raw"])

    args = parser.parse_args()

    main(args)
