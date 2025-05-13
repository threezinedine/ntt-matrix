import os
import argparse
import numpy as np
from jinja2 import Template

parser = argparse.ArgumentParser()
parser.add_argument("input", type=str)
args = parser.parse_args()


template_string = """
ntt::Tensor {{name}} = ntt::Tensor::from_vector(
    {{type}}{{data}}
);
"""

data = np.load(args.input)
input_file_name_only = os.path.basename(args.input).split(".")[0]
len_of_shape = len(data.shape)
data_list = data.tolist()
new_data = str(data_list).replace("[", "{").replace("]", "}")

if len_of_shape == 1:
    type_data = "ntt::vec"
elif len_of_shape == 2:
    type_data = "ntt::tensor2d"
elif len_of_shape == 3:
    type_data = "ntt::tensor3d"
else:
    type_data = "ntt::tensor4d"


template = Template(template_string)
result = template.render(data=new_data, name=input_file_name_only, type=type_data)

output_file_name = f"{input_file_name_only}.tasm"
with open(output_file_name, "w") as f:
    f.write(result)

output_binary_file_name = f"{input_file_name_only}.bin"

bin_data = [len_of_shape.to_bytes(1, "big")]
for i in range(len_of_shape):
    bin_data.append(data.shape[i].to_bytes(8, "little"))

bin_data.append(data.flatten().tobytes())
bin_data = b"".join(bin_data)

with open(output_binary_file_name, "wb") as f:
    f.write(bin_data)
