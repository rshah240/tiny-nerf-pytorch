import torch
from model import VeryTinyNerfModel
from utils import *
from tqdm import tqdm
from PIL import Image
import os
import numpy as np

# Chunksize (Note: this isn't batchsize in the conventional sense. This only
# specifies the number of rays to be queried in one go. Backprop still happens
# only after all rays from the current "bundle" are queried and rendered).
chunksize = 4096  # Use chunksize of about 4096 to fit in ~1.4 GB of GPU memory.

def get_data():
    """
    Function to get the data
    """
    data_path = "data/tiny_nerf_data.npz"
    data = np.load(data_path)

    return data



# One iteration of TinyNeRF (forward pass).
def run_one_iter_of_tinynerf(height, width, focal_length, tform_cam2world,
                            near_thresh, far_thresh, depth_samples_per_ray,
                            encoding_function, get_minibatches_function, model):

    # Get the "bundle" of rays through all image pixels.
    ray_origins, ray_directions = get_ray_bundle(height, width, focal_length,
                                                tform_cam2world)
    
    # Sample query points along each ray
    query_points, depth_values = compute_query_points_from_rays(
        ray_origins, ray_directions, near_thresh, far_thresh, depth_samples_per_ray
    )

    # "Flatten" the query points.
    flattened_query_points = query_points.reshape((-1, 3))

    # Encode the query points (default: positional encoding).
    encoded_points = encoding_function(flattened_query_points)

    # Split the encoded points into "chunks", run the model on all chunks, and
    # concatenate the results (to avoid out-of-memory issues).
    batches = get_minibatches_function(encoded_points, chunksize=chunksize)
    predictions = []
    for batch in batches:
        predictions.append(model(batch))
    radiance_field_flattened = torch.cat(predictions, dim=0)

    # "Unflatten" to obtain the radiance field.
    unflattened_shape = list(query_points.shape[:-1]) + [4]
    radiance_field = torch.reshape(radiance_field_flattened, unflattened_shape)

    # Perform differentiable volume rendering to re-synthesize the RGB image.
    rgb_predicted, _, _ = render_volume_density(radiance_field, ray_origins, depth_values)

    return rgb_predicted


def eval():
    """
    Function to eval the model on lego dataset
    """
    # Images
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    data = get_data()
    images = data["images"]
    # Camera extrinsics (poses)
    tform_cam2world = data["poses"]
    tform_cam2world = torch.from_numpy(tform_cam2world).to(device)
    # Focal length (intrinsics)
    focal_length = data["focal"]
    focal_length = torch.from_numpy(focal_length).to(device)

    # Height and width of each image
    height, width = images.shape[1:3]

    # Near and far clipping thresholds for depth values.
    near_thresh = 2.
    far_thresh = 6.

    # Seed RNG, for repeatability
    seed = 9458
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Specify encoding function.
    num_encoding_functions = 6
    encode = lambda x: positional_encoding(x, num_encoding_functions=num_encoding_functions)
    # Number of depth samples along each ray.
    depth_samples_per_ray = 32

    checkpoint_path = "checkpoints/nerf_exp1.pth"
    config = torch.load(checkpoint_path)
    model = VeryTinyNerfModel()
    model.load_state_dict(config["state_dict"])
    model.to(device)
    model.eval()

    number_images = images.shape[0]
    with torch.no_grad():
        for i in tqdm(range(number_images)):
            target_tform_cam2world = tform_cam2world[i].to(device)
            rgb_predicted = run_one_iter_of_tinynerf(height, width, focal_length,
                            target_tform_cam2world, near_thresh,
                            far_thresh, depth_samples_per_ray,
                            encode, get_minibatches, model)
            image = Image.fromarray((rgb_predicted.detach().cpu().numpy()*255).astype(np.uint8))
            image_path = os.path.join("results", str(i) + ".png")
            image.save(image_path)


if __name__ == "__main__":
    eval()

        