import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

import numpy as np
from socket import timeout
import urllib
# from urllib.request import Request, urlopen  # Python 3
from urllib.error import HTTPError, URLError


def get_image_from_url(url):
    try:
        # q = Request(url)
        # agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'
        # q.add_header('User-Agent', agent)
        # response = urlopen(q)
        response = urllib.request.urlopen(url, timeout=10, headers='abc')
    except (HTTPError, URLError):
        return None
    except timeout:
        return None
    else:
        # print('Access successful.')
        return plt.imread(response, 'jpg')


def image_scatter(X, images, ax=None, zoom=1, highlight=False):
    if ax is None:
        ax = plt.gca()

    for i in range(X.shape[0]):
        im = OffsetImage(images[i], zoom=zoom)
        if highlight:
            ab = AnnotationBbox(im, (X[i, 0], X[i, 1]), xycoords='data', frameon=True, bboxprops=dict(edgecolor='red'))
        else:
            ab = AnnotationBbox(im, (X[i, 0], X[i, 1]), xycoords='data', frameon=False)
        ax.add_artist(ab)
    ax.update_datalim(np.column_stack([X[:, 0], X[:, 1]]))
    ax.autoscale()


def get_all_images(all_urls):
    all_images = []
    for i in range(len(all_urls)):
        img = get_image_from_url(all_urls[i])
        if img is not None:
            all_images.append(img)
            if len(all_images) % 100 == 0:
                print("Processed: {}".format(len(all_images)))
        else:
            print("Invalid image at {}".format(i))
    return all_images
