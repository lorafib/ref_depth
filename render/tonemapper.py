
import torch
from . import util
import numpy as np

class ToneMapper(torch.nn.Module):
    """Tone mapping module capable of learning per-camera exposure, and response."""

    def __init__(self, response_mode="constant", from_path= None) -> None:
        """Initialize submodules."""
        super().__init__()
        self.response_mode = response_mode
        # init response params
        if response_mode == "piecewise":
            response = torch.linspace(0.0, 1.0, 25).pow(0.4545454681)
            response = response / response[-1]
            response = response[None, None, None, :].repeat(1, 3, 1, 1)
        elif response_mode == "constant":
            response = torch.ones((1,3,1,1))
        else: 
            print("invalid response mode")
            exit()
        
        if from_path != None:
            response = torch.tensor(np.loadtxt(from_path), dtype=torch.float32).reshape(1,3,1,1)
            
        self.register_parameter('response_params', torch.nn.Parameter(response))

    def setup_exposure_params(self, n_cameras:float, from_path= None) -> None:
        """Set up the exposure parameters."""
        exposure_params = torch.zeros(n_cameras)
        
        if from_path != None:
            exposure_params = torch.tensor(np.loadtxt(from_path), dtype=torch.float32)
        
        timestamp2index = torch.tensor(n_cameras - 1)
        # init params
        self.register_parameter('exposure_params', torch.nn.Parameter(exposure_params))
        self.register_buffer('timestamp2index', timestamp2index)

    def apply_exposure(self, image: torch.Tensor, timestamp: float) -> torch.Tensor:
        """Apply exposure correction to the image."""
        camera_idx = (self.timestamp2index * timestamp).to(dtype=torch.long)
        return image / 2 ** self.exposure_params[camera_idx]

    def apply_response_piecewise_linear(self, image: torch.Tensor, train_mode: bool) -> torch.Tensor:
        """Apply response correction to the image."""
        image = image.squeeze()
        leak_add = None
        if train_mode:
            clamp_low = image < 0.0
            clamp_high = image > 1.0
            leak_add = (image * 0.0099999998) * clamp_low
            leak_add += (-0.0099999998 / image.abs().add(1.0e-4).sqrt() + 0.0099999998) * clamp_high
        image = image * 2.0 - 1.0
        x = torch.stack([image, torch.zeros_like(image)], dim=-1)
        result = torch.empty_like(image)
        # TODO: remove loop
        for ch in range(3):
            result[ch] = torch.nn.functional.grid_sample(
                input=self.response_params[:, ch:ch + 1], grid=x[ch:ch + 1],
                align_corners=True, mode='bilinear', padding_mode='border'
            ).squeeze()
        if train_mode:
            result += leak_add
        # image = image.permute(2,0,1).unsqueeze(0)
        result = result.unsqueeze(0)
        return result
    
    def apply_response_gamma(self, image: torch.Tensor, train_mode: bool) -> torch.Tensor:
        
        # apply channel-wise response
        image = torch.pow(image, self.response_params) 
        
        # to standard srgb
        srgb = util._rgb_to_srgb(image)
        
        
        if torch.any(srgb.isnan()):
            print(self.response_params, "apply")
        # else: print("not yet nan")
        
        return srgb

    def apply_response(self, image: torch.Tensor, train_mode: bool) -> torch.Tensor:
        if self.response_mode == "piecewise":
            return self.apply_response_piecewise_linear(image, train_mode)
        elif self.response_mode == "constant":
            return self.apply_response_gamma(image, train_mode)
        else: 
            print("invalid response mode")
            exit()

    def get_optimizer_param_groups(self, max_iterations: int): # -> tuple[list[dict], list[LRDecayPolicy]]:
        """Returns the parameter groups for the optimizer."""
        param_groups = [
            {'params': [self.exposure_params], 'lr': 5.0e-3},
            {'params': [self.response_params], 'lr': 5.0e-4},
        ]
        
        return param_groups #, schedulers

    def calc_response_regularizer(self) -> torch.Tensor:
        """Calculate the response regularizer."""
        return torch.nn.functional.mse_loss(self.response_params, torch.ones_like(self.response_mode), reduction='sum')

    def forward(self, image: torch.Tensor, camera_timestamp: float, train_mode: bool) -> torch.Tensor:
        """Apply tone mapping to the input image."""
        # we use timestamp because our GUI allows to change it, thus we can modify the exposure in the GUI
        image = self.apply_exposure(image, camera_timestamp) # camera_timestamp in [0,1]
        image = self.apply_response(image, train_mode)
        return image

class InvertableToneMapper(ToneMapper):
    
    def check_monotonicity(values: torch.Tensor) -> tuple[bool, bool]:
        """
        Check if the given 1D values tensor is monotonic.
        Returns:
            (is_monotonic: bool, increasing: bool)
        """
        if values.dim() != 1:
            raise ValueError("check_monotonicity expects a 1D tensor.")

        # Compute differences
        diffs = values[1:] - values[:-1]

        # Check if all diffs are >= 0 or all diffs are <= 0
        non_decreasing = torch.all(diffs >= 0)
        non_increasing = torch.all(diffs <= 0)

        if non_decreasing and not non_increasing:
            return True, True  # monotonic increasing
        elif non_increasing and not non_decreasing:
            return True, False  # monotonic decreasing
        elif non_increasing and non_decreasing:
            # This means all diffs are zero => it's a constant array (also monotonic)
            return True, True  # can consider constant as non-decreasing
        else:
            # not monotonic
            return False, False

    
    def inverse_exposure(self, image: torch.Tensor, timestamp: float) -> torch.Tensor:
        """Inverse of the exposure correction."""
        camera_idx = int(self.timestamp2index * timestamp)
        return image * 2 ** self.exposure_params[camera_idx]

    def inverse_response_piecewise_linear(self, result: torch.Tensor, train_mode: bool) -> torch.Tensor:
        """
        Invert the response applied by apply_response, returning an estimate of the original image.

        Assumptions:
        - self.response_params has shape (1, 3, 1, 25).
        This represents a LUT for each channel, sampled at 25 points from 0 to 1 (then remapped to [-1,1]).
        - The result has shape (1, 3, H, W).
        - The mapping is channel-wise and independent for each pixel.
        - We assume monotonicity in the response function for proper inversion.

        If train_mode was applied, no perfect analytic inverse is provided here for the leak term.
        """

        # result shape: (1,3,H,W)
        # Remove the batch dimension for convenience
        result = result.squeeze(0)  # (3,H,W)

        # response_params shape is (1,3,1,25)
        # We want to get it into shape (3,25) so we can handle each channel as a 1D array
        params = self.response_params.squeeze(0).squeeze(1)  # now (3,25)

        # Number of samples along the LUT width dimension
        N = params.shape[1]
        # Create the domain vector for the LUT: N samples from -1 to 1
        x_domain = torch.linspace(-1.0, 1.0, steps=N, device=params.device, dtype=params.dtype)

        def invert_channel(channel_values, target):
            # channel_values: shape (N,) monotonic array for one channel
            # target: (H,W) tensor containing the values to invert
            # returns (H,W) tensor of x in [-1,1] that maps to target by interpolation
            original_shape = target.shape
            target = target.flatten()

            # Check monotonicity: assume increasing if last > first
            increasing = (channel_values[-1] > channel_values[0])

            # Binary search to find interval
            left = torch.zeros_like(target, dtype=torch.long, device=channel_values.device)
            right = (N - 1) * torch.ones_like(target, dtype=torch.long, device=channel_values.device)

            # Perform approximately log2(N) steps
            steps = int(torch.log2(torch.tensor(N, dtype=torch.float)).ceil().item()) + 1
            for _ in range(steps):
                mid = (left + right) // 2
                mid_val = channel_values[mid]
                if increasing:
                    cond = mid_val < target
                else:
                    cond = mid_val > target
                left = torch.where(cond, mid, left)
                right = torch.where(cond, right, mid)

            # Clamp left to a valid interval [0, N-2]
            left = torch.clamp(left, 0, N - 2)
            right = left + 1

            # Linear interpolation for finer solution
            left_x = x_domain[left]
            right_x = x_domain[right]
            left_val = channel_values[left]
            right_val = channel_values[right]

            # Avoid division by zero if values are equal
            denom = (right_val - left_val)
            denom[denom.abs() < 1e-10] = 1e-10

            frac = (target - left_val) / denom
            x_est = left_x + frac * (right_x - left_x)

            return x_est.view(original_shape)

        # Invert each channel
        inv_image = torch.empty_like(result)  # (3,H,W)
        for ch in range(3):
            inv_image[ch] = invert_channel(params[ch], result[ch])

        # Now inv_image is in [-1,1], convert back to [0,1]
        inv_image = (inv_image + 1.0) / 2.0

        if train_mode:
            # If we must handle leak correction accurately, we'd need iterative methods.
            # For now, we leave this part out or just note that this is a simplified inversion.
            pass

        return inv_image.unsqueeze(0)  # add the batch dimension back

    def inverse_response_gamma(self, result: torch.Tensor, train_mode: bool) -> torch.Tensor:
        # from standard srgb
        image = util._srgb_to_rgb(result)
                
        # invert channel-wise response, beware of negative bases!
        image = torch.pow(torch.clamp_min(image, 0.0001), torch.clamp_min(1.0/self.response_params, 0.0001)) 
        
        if torch.any(image.isnan()):
            print(self.response_params, "inverse")
        # else: print("not yet nan")
        
        return image
    
    def inverse_response(self, result: torch.Tensor, train_mode: bool) -> torch.Tensor:
        if self.response_mode == "piecewise":
            return self.inverse_response_piecewise_linear(result, train_mode)
        elif self.response_mode == "constant":
            return self.inverse_response_gamma(result, train_mode)
        else: 
            print("invalid response mode")
            exit()
            
    def inverse_forward(self, image: torch.Tensor, camera_timestamp: float, train_mode: bool) -> torch.Tensor:
        """Apply inverse tone mapping to the input image."""
        image = self.inverse_response(image, train_mode)
        image = self.inverse_exposure(image, camera_timestamp)  # camera_timestamp in [0,1]
        return image



import torch
import os
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image

@torch.no_grad()
def test_case_1():
    """Test case 1: Process four white images using the ToneMapper module."""
    # Initialize ToneMapper
    tonemapper = ToneMapper()
    tonemapper.setup_exposure_params(n_cameras=4)

    # Create four white images of size 10x10
    white_images = [0.25*torch.ones(1, 3, 10, 10) for _ in range(4)]

    # Process and print results
    print("Test Case 1 Results:")
    for i, img in enumerate(white_images):
        result = tonemapper(img, camera_timestamp=i*(1.0/4), train_mode=False)
        print(f"Image {i + 1} Result:\n{result.squeeze().detach().cpu().numpy()}\n")

@torch.no_grad()
def test_case_2(directory_path):
    """Test case 2: Load all images from a directory, process them with ToneMapper, and plot results."""
    # Load images from the directory
    images = []
    image_names = []
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff')):
            img_path = os.path.join(directory_path, filename)
            image = Image.open(img_path).convert('RGB')
            images.append(ToTensor()(image).unsqueeze(0))  # Convert to tensor and add batch dimension
            image_names.append(filename)
        if len(image_names) > 4: break

    # Initialize ToneMapper with n_cameras equal to the number of images
    tonemapper = ToneMapper()
    tonemapper.setup_exposure_params(n_cameras=len(images))

    # Process images
    processed_images = []
    for i, img in enumerate(images):
        processed_img = tonemapper(img, camera_timestamp=i*(1.0/len(images)), train_mode=False)
        processed_images.append(processed_img.squeeze(0))  # Remove batch dimension

    # Plot results
    fig, axes = plt.subplots(1, len(processed_images), figsize=(15, 5))
    for i, (original, processed) in enumerate(zip(images, processed_images)):
        ax = axes[i]
        ax.imshow(ToPILImage()(processed.clamp(0, 1)))  # Clamp values to [0, 1] for visualization
        ax.set_title(image_names[i])
        ax.axis('off')
    plt.show()

def test_case_3():
    """Test case 3: Test inverse tone mapping on test_case_1 inputs."""
    # Initialize ToneMapper
    tonemapper = InvertableToneMapper()
    tonemapper.setup_exposure_params(n_cameras=1)

    # Create four white images of size 10x10
    white_images = [torch.ones(1, 3, 10, 10) for _ in range(4)]

    # Process, inverse, and reapply tone mapping
    print("Test Case 3 Results:")
    for i, img in enumerate(white_images):
        inverted = tonemapper.inverse_forward(img, camera_timestamp=i*(1.0/4), train_mode=False)
        reapplied = tonemapper(inverted, camera_timestamp=i*(1.0/4), train_mode=False)
        diff = (img - reapplied).abs().mean()
        print(f"Image {i + 1} Difference: {diff.item()}\n")

def test_case_4(directory_path):
    """Test case 4: Test inverse tone mapping on test_case_2 inputs."""
    # Load images from the directory
    images = []
    image_names = []
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff')):
            img_path = os.path.join(directory_path, filename)
            image = Image.open(img_path).convert('RGB')
            images.append(ToTensor()(image).unsqueeze(0))  # Convert to tensor and add batch dimension
            image_names.append(filename)
        if len(image_names) > 4: break

    # Initialize ToneMapper with n_cameras equal to the number of images
    tonemapper = InvertableToneMapper()
    tonemapper.setup_exposure_params(n_cameras=len(images))

    # Process, inverse, and reapply tone mapping
    processed_images = []
    diff_images = []
    for i, img in enumerate(images):
        inverted = tonemapper.inverse_forward(img, camera_timestamp=i*(1.0/len(images)), train_mode=False)
        reapplied = tonemapper(inverted, camera_timestamp=i*(1.0/len(images)), train_mode=False)
        diff = (img - reapplied).abs()
        processed_images.append(reapplied.squeeze(0))
        diff_images.append(diff.squeeze(0))

    # Plot results
    fig, axes = plt.subplots(2, len(processed_images), figsize=(15, 10))
    for i, (processed, diff, name) in enumerate(zip(processed_images, diff_images, image_names)):
        ax1 = axes[0, i]
        ax1.imshow(ToPILImage()(processed.clamp(0, 1)))  # Clamp values to [0, 1] for visualization
        ax1.set_title(f"Reapplied {name}")
        ax1.axis('off')

        ax2 = axes[1, i]
        ax2.imshow(ToPILImage()(diff.clamp(0, 1)))  # Clamp values to [0, 1] for visualization
        ax2.set_title(f"Difference {name}")
        ax2.axis('off')
    plt.show()

if __name__ == "__main__":
    # Run test case 1
    test_case_1()

    # Run test case 2
    directory_path = "/home/qe37qulu/repos/depth-refinement/synth"  # Replace with your directory path
    if os.path.exists(directory_path):
        # test_case_2(directory_path)
        test_case_4(directory_path)
    else:
        print(f"Directory {directory_path} does not exist. Please provide a valid path.")

    # Run test case 3
    test_case_3()