import os
import re

def_results_base_dir = "configurations/results/"
def_results_dir = 'results/'
def_cfg_file = 'configurations/configs.cfg'
def_delim = '='

def_bvae_out_dir = "block_vae_out/"
def_bcnn_out_dir = "block_cnn_out/"

def is_float(element):
    if not re.match("^\d+?\.\d+?$", element) is None:
       return True
    return False

def create_dir_if_not_already(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

class Configs:
    """
    This class will get a series of parameters from a given
    configuration file and make them be their attributes.

    Args:
        cfg_file (str): name of the configuration file.
        delim (str): Delimiter between each variable name and
            its value in the configuration file.

    Attributes:
        The class will automatically set its attributes to be
        the ones in the configuration file. To access them just
        need to write self.<name_var_in_cfg_file>

        cfg_vals (dict): Dictionary where all the config
            key and values will be stored

        cfg_file (str): Name of the config file where we are
            reading from
    """

    def read_config_file(self, file=def_cfg_file, delim=def_delim):
        with open(file, 'r') as f:
            for line in f:
                line = line.rstrip().replace(" ", "")

                if (line == "\n") or ('#' in line) or (not line):
                    continue

                name, val = line.split(delim)
                val = int(val) if val.isdigit() else val
                val = float(val) if is_float(str(val)) else val
                self.cfg_vals[name] = val

    def __init__(self, cfg_file=def_cfg_file, delim=def_delim,
                 bvae_out_dir=def_bvae_out_dir, bcnn_out_dir=def_bcnn_out_dir):
        self.cfg_vals = {}
        self.cfg_file = cfg_file
        self.results_base_dir = def_results_base_dir
        self.results_dir = ""
        self.block_vae_outputs_dir = bvae_out_dir
        self.block_cnn_outputs_dir = bcnn_out_dir

        self.read_config_file(cfg_file, delim)

        # Set as new self parameters the values extracted
        # from the configuration file. To access their values
        # we need to write self.<name of cfg var>
        for key, val in self.cfg_vals.items():
            setattr(self, key, val)

        self.results_dir = self.results_dir if self.results_dir != "" \
                                                        else def_results_dir

        create_dir_if_not_already(self.results_dir)
        create_dir_if_not_already(self.results_dir+self.block_vae_outputs_dir)
        create_dir_if_not_already(self.results_dir+self.block_cnn_outputs_dir)

    def get_bcnn_out_path(self):
        return self.results_dir+self.block_cnn_outputs_dir

    def get_bvae_out_path(self):
        return self.results_dir+self.block_vae_outputs_dir