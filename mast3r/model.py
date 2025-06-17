import torch
import torch.nn.functional as F
import os

from mast3r.catmlp_dpt_head import mast3r_head_factory
import mast3r.utils.path_to_dust3r  # noqa
from dust3r.model import AsymmetricCroCo3DStereo  # noqa
from dust3r.utils.misc import transpose_to_landscape  # noqa

inf = float('inf')

def load_model(model_path, device, verbose=True):
    """
    Load a model from the given path.

    Args:
        model_path (str): The path to the model checkpoint.
        device (torch.device): The device to load the model onto.
        verbose (bool): Whether to print loading details. Default is True.

    Returns:
        torch.nn.Module: The loaded model.
    """
    if verbose:
        print('... loading model from', model_path)
    ckpt = torch.load(model_path, map_location='cpu')
    args = ckpt['args'].model.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")

    # Ensure landscape_only is set to False
    if 'landscape_only' not in args:
        args = args[:-1] + ', landscape_only=False)'
    else:
        args = args.replace(" ", "").replace('landscape_only=True', 'landscape_only=False')

    assert "landscape_only=False" in args

    if verbose:
        print(f"instantiating : {args}")

    net = eval(args)
    s = net.load_state_dict(ckpt['model'], strict=False)

    if verbose:
        print(s)

    return net.to(device)

class AsymmetricMASt3R(AsymmetricCroCo3DStereo):
    """
    AsymmetricMASt3R model class.
    """

    def __init__(self, desc_mode=('norm'), two_confs=False, desc_conf_mode=None, **kwargs):
        """
        Initialize the AsymmetricMASt3R model.

        Args:
            desc_mode (tuple): Description mode. Default is ('norm').
            two_confs (bool): Whether to use two configurations. Default is False.
            desc_conf_mode (str): Description configuration mode. Default is None.
            **kwargs: Additional keyword arguments for the parent class.
        """
        self.desc_mode = desc_mode
        self.two_confs = two_confs
        self.desc_conf_mode = desc_conf_mode
        super().__init__(**kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        """
        Load a pretrained model.

        Args:
            pretrained_model_name_or_path (str): Path to the pretrained model or model name.
            **kw: Additional keyword arguments for loading the model.

        Returns:
            AsymmetricMASt3R: The loaded model.
        """
        if os.path.isfile(pretrained_model_name_or_path):
            return load_model(pretrained_model_name_or_path, device='cpu')
        else:
            return super(AsymmetricMASt3R, cls).from_pretrained(pretrained_model_name_or_path, **kw)

    def set_downstream_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode, patch_size, img_size, **kw):
        """
        Set the downstream head for the model.

        Args:
            output_mode (str): Output mode.
            head_type (str): Head type.
            landscape_only (bool): Whether to only use landscape orientation.
            depth_mode (str): Depth mode.
            conf_mode (str): Confidence mode.
            patch_size (int): Patch size.
            img_size (tuple): Image size.
            **kw: Additional keyword arguments.
        """
        assert img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0, \
            f'Image size {img_size} must be a multiple of patch size {patch_size}'

        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode

        if self.desc_conf_mode is None:
            self.desc_conf_mode = conf_mode

        # Allocate heads
        self.downstream_head1 = mast3r_head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        self.downstream_head2 = mast3r_head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))

        # Magic wrapper
        self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)
        self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)
