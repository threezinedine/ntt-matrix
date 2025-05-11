import os
import argparse
import numpy as np
from jinja2 import Template

parser = argparse.ArgumentParser()
parser.add_argument("input", type=str)
parser.add_argument("name", type=str)
args = parser.parse_args()

template_string = """
ntt::Tensor {{name}} = ntt::Tensor::from_vector(
    {{data}}
);
"""

data = np.load(args.input)
data = data.tolist()
new_data = str(data).replace("[", "{").replace("]", "}")

template = Template(template_string)
result = template.render(data=new_data, name=args.name)

output_file_name = f"{args.name}.tasm"
with open(output_file_name, "w") as f:
    f.write(result)
