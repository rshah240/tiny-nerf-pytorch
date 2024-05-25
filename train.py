import torch
from model import VeryTinyNerfModel
from utils import *
from torch.nn import MSELoss
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

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


def train():

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

    # Hold one image out (for test).
    testimg, testpose = images[101], tform_cam2world[101]
    testimg = torch.from_numpy(testimg).to(device)

    # Map images to device
    images = torch.from_numpy(images[:100, ..., :3]).to(device)

    # Number of functions used in the positional encoding (Be sure to update the 
    # model if this number changes).
    num_encoding_functions = 6
    # Specify encoding function.
    encode = lambda x: positional_encoding(x, num_encoding_functions=num_encoding_functions)
    # Number of depth samples along each ray.
    depth_samples_per_ray = 32

    # Optimizer parameters
    lr = 5e-3
    num_iters = 1000

    # Misc parameters
    display_every = 100  # Number of iters after which stats are displayed
    model = VeryTinyNerfModel()
    model.to(device)

    iter_epochs = tqdm(range(num_iters))


    optimizer = Adam(model.parameters(), lr=lr)
    criterion = MSELoss()
    
    # Seed RNG, for repeatability
    seed = 9458
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Lists to log metrics etc.
    psnrs = []
    iternums = []

    writer = SummaryWriter("logs/nerf_exp_1")
    
    for i in iter_epochs:

        optimizer.zero_grad()
        # Randomly pick an image as the target.
        target_img_idx = np.random.randint(images.shape[0])
        target_img = images[target_img_idx].to(device)
        target_tform_cam2world = tform_cam2world[target_img_idx].to(device)

        # Run one iteration of TinyNeRF and get the rendered RGB image.
        rgb_predicted = run_one_iter_of_tinynerf(height, width, focal_length,
                                                target_tform_cam2world, near_thresh,
                                                far_thresh, depth_samples_per_ray,
                                                encode, get_minibatches, model)
        

        loss = criterion(rgb_predicted, target_img)
        loss.backward()
        optimizer.step()

        writer.add_scalar("Nerf Loss", loss.item(), i)

        if i % display_every == 0:
            # Render the held-out view
            rgb_predicted = run_one_iter_of_tinynerf(height, width, focal_length,
                                                    testpose, near_thresh,
                                                    far_thresh, depth_samples_per_ray,
                                                    encode, get_minibatches, model)
            loss = criterion(rgb_predicted, target_img)
            psnr = -10. * torch.log10(loss)
            
            psnrs.append(psnr.item())
            iternums.append(i)

            figure = plt.figure(figsize = (10,4))
            plt.subplot(121)
            plt.imshow(rgb_predicted.detach().cpu().numpy())
            plt.title(f"Iteration {i}")
            plt.subplot(122)
            plt.plot(iternums, psnrs)
            plt.title("PSNR")

            writer.add_figure("Generated Output", figure, i)

    print("Saving the Model")
    path = "checkpoints/nerf_exp1.pth"
    config = {}
    config["state_dict"] = model.state_dict()
    torch.save(config, path)


if __name__ == "__main__":
    train()




        



    

