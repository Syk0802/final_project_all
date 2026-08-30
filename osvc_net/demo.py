import argparse
import os

import matplotlib
import numpy as np
import scipy.io
import torch

matplotlib.use("agg")
import matplotlib.pyplot as plt


#####################################################################
# Show result
def imshow_with_border(ax, path, border_color=None, border_lw=3):
    """Show image on given axes, optionally draw a colored border around it."""
    im = plt.imread(path)
    ax.imshow(im)
    ax.set_xticks([])
    ax.set_yticks([])
    if border_color is None:
        # hide spines completely (for the query image)
        for s in ax.spines.values():
            s.set_visible(False)
    else:
        # use spines as the colored bounding box, sitting tight on the image
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_edgecolor(border_color)
            s.set_linewidth(border_lw)


def sort_img(qf, ql, qc, gf, gl, gc):
    query = qf.view(-1, 1)
    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]
    # good index
    query_index = np.argwhere(gl == ql)
    # same camera
    camera_index = np.argwhere(gc == qc)

    junk_index1 = np.argwhere(gl == -1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1)

    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]
    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--query_index", default=-1, type=int, help="test_image_index")
    parser.add_argument("--nums", default=100, type=int, help="need nums query to rank")
    parser.add_argument(
        "--output", default="./ranklist", type=str, help="save ranklist figure"
    )
    parser.add_argument(
        "--mat_file", default="pytorch_result.mat", type=str, help="path to .mat result file"
    )
    parser.add_argument("--topk", default=10, type=int, help="top-k gallery images to show")
    parser.add_argument("--num_query", default=10, type=int, help="number of query samples to visualize")
    opts = parser.parse_args()

    os.makedirs(opts.output, exist_ok=True)

    result = scipy.io.loadmat(opts.mat_file)
    query_feature = torch.FloatTensor(result["query_f"])
    query_cam = result["query_cam"][0]
    query_label = result["query_label"][0]
    query_img_paths = result["query_img_paths"]
    gallery_feature = torch.FloatTensor(result["gallery_f"])
    gallery_cam = result["gallery_cam"][0]
    gallery_label = result["gallery_label"][0]
    gallery_img_paths = result["gallery_img_paths"]

    query_feature = query_feature.cuda()
    gallery_feature = gallery_feature.cuda()

    topk = opts.topk

    # Layout: one row per query, each image cell ~ 128x256 (W x H) for market1501.
    cell_w_in = 1.2   # inches per image column
    cell_h_in = 2.4   # inches per image row (person images are tall)
    # Total columns = 1 (query) + topk (gallery); use the same wspace for all gaps.
    n_cols = 1 + topk
    fig_w_in = cell_w_in * n_cols

    for i in range(opts.num_query):
        index = sort_img(
            query_feature[i],
            query_label[i],
            query_cam[i],
            gallery_feature,
            gallery_label,
            gallery_cam,
        )

        query_path = query_img_paths[i].strip()
        cur_query_label = query_label[i]
        print(query_path)
        print(f"Top {topk} images are as follow:")

        try:
            fig = plt.figure(figsize=(fig_w_in, cell_h_in))
            gs = fig.add_gridspec(
                1,
                n_cols,
                left=0,
                right=1,
                top=1,
                bottom=0,
                wspace=0.08,  # uniform gap between all images (query & gallery)
                hspace=0,
            )

            # query image (col 0), no colored border
            ax_q = fig.add_subplot(gs[0, 0])
            imshow_with_border(ax_q, query_path, border_color=None)

            for j in range(topk):
                ax = fig.add_subplot(gs[0, 1 + j])
                img_path = gallery_img_paths[index[j]].strip()
                label = gallery_label[index[j]]
                color = "#00B050" if label == cur_query_label else "#E10600"
                imshow_with_border(ax, img_path, border_color=color, border_lw=3)
                print(img_path)

            out_path = os.path.join(opts.output, f"{i}.png")
            # bbox_inches='tight' + pad_inches=0 removes remaining margins
            fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.02)
            plt.close(fig)
        except RuntimeError:
            print("Skip!!!")
            plt.close("all")
