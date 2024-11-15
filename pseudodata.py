from PIL import Image
import glob
import numpy as np

all_label_path=glob.glob('D:\\train\\train\\label\\*.png')
label=Image.open(all_label_path[0])
label=np.array(label)
label=label/100
label=label.astype(np.int64)
print(np.unique(label))