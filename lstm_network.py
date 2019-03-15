import numpy as np 
import tensorflow as tf 
from utils import *
from tensorflow.keras import layers 
from models import *

name = "Lazarus"
sequence_length = 80
data = "/Users/thomasklein/Projects/BremenBigDataChallenge2019/bigdatachallenge/data/rawdata.tfrecords"
model_path = "/Users/thomasklein/Projects/BremenBigDataChallenge2019/bigdatachallenge/models/"
archive_path = model_path + "model_archive/"

modus = 'weird' 

model = get_lazarus(sequence_length)

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

if modus == 'train':

    np.random.seed(42)
    indices = np.random.randint(0, 6384, 638)
    dataset = read_recurrent_dataset(data, sequence_length, filter_ids=indices, mode='exclude')
    validation_set = read_recurrent_dataset(data, sequence_length, filter_ids=indices, mode='include')

elif modus == 'subject_train':

    indices = [18,19]
    dataset = read_recurrent_dataset(data, sequence_length, filter_subjects=indices, mode='exclude')
    validation_set = read_recurrent_dataset(data, sequence_length, filter_subjects=indices, mode='include')


elif modus == 'full':

    dataset = read_recurrent_dataset(data, sequence_length)
    validation_set = dataset

elif modus == 'weird':
    olddata = "/Users/thomasklein/Projects/BremenBigDataChallenge2019/bigdatachallenge/data/old_rawdata.tfrecords"
    indices = [2881, 725, 4204, 1894, 2827, 574, 2187, 2310, 1223, 4106, 1360, 106, 1193, 2707, 3846, 2007, 1838, 5693, 4660, 5363, 1557, 3569, 776, 1586, 4716, 4417, 6182, 5597, 1024, 2441, 6069, 5296, 
    297, 3779, 5232, 4885, 67, 4076, 1370, 6139, 8, 900, 3833, 1747, 2738, 3843, 297, 4673, 4211, 2322, 2391, 3881, 1634, 297, 4920, 1904, 869, 1035, 5087, 5651, 1981, 1954, 421, 2429, 1706, 5883, 5937, 
    6215, 5980, 459, 1822, 673, 6134, 3925, 1829, 1393, 3765, 2017, 2717, 40, 4223, 3245, 3722, 5378, 4434, 1423, 471, 4699, 1871, 4991, 4419, 4584, 5970, 877, 5684, 631, 50, 260, 5874, 1207, 2123, 1352, 
    4679, 3904, 5884, 4058, 6090, 4483, 651, 5190, 451, 3169, 4145, 2801, 6175, 2185, 3708, 3623, 543, 36, 4407, 3160, 2769, 4791, 5873, 91, 1601, 5200, 4915, 4697, 5676, 1315, 1148, 993, 1557, 933, 1779, 
    2983, 1319, 5682, 2113, 2808, 5517, 3831, 1762, 474, 5429, 4634, 2642, 4641, 4207, 252, 908, 2152, 176, 5750, 6105, 5148, 3188, 436, 314, 5611, 2616, 4660, 4620, 4769, 2853, 3436, 3664, 238, 757, 852, 
    6317, 5717, 4027, 2561, 4050, 4505, 3982, 5136, 601, 1316, 2399, 338, 1507, 4252, 653, 214, 3900, 4561, 774, 5204, 3048, 2076, 6378, 5226, 1258, 5170, 1411, 2912, 3905, 3774, 1535, 5231, 5663, 1338, 6140, 
    5276, 2294, 1911, 1000, 4767, 4277, 4894, 387, 1334, 2919, 3704, 967, 1103, 320, 4729, 3101, 440, 1772, 3673, 1879, 6341, 4019, 6285, 3945, 6144, 4284, 6251, 5615, 235, 1548, 977, 1993, 1003, 2415, 3081, 
    5619, 4890, 5548, 207, 606, 3995, 3627, 169, 5520, 5667, 2336, 1681, 3451, 3640, 5370, 1685, 2001, 742, 2801, 5920, 4487, 3796, 5218, 5718, 4888, 1398, 40, 5815, 5635, 6288, 433, 3885, 5733, 1732, 1027, 
    1435, 3019, 6012, 6271, 1907, 4249, 5621, 4326, 2004, 2712, 2835, 3219, 4015, 913, 1351, 5312, 4046, 2574, 245, 3974, 5505, 343, 3301, 6012, 4892, 1240, 1174, 4781, 5996, 2055, 1292, 756, 3916, 5118, 1532, 
    3403, 5978, 4386, 6035, 3581, 1220, 3662, 2285, 1586, 1331, 2212, 1423, 825, 5446, 2137, 2188, 5980, 5753, 233, 114, 678, 3386, 3002, 4626, 890, 6318, 4232, 2953, 5607, 4767, 4448, 2387, 4238, 2669, 3827, 
    654, 4457, 537, 748, 3553, 5374, 3196, 3098, 4245, 2761, 5444, 1646, 4372, 6126, 135, 3795, 3130, 582, 2114, 3234, 1336, 3365, 469, 5057, 953, 5721, 855, 5315, 1476, 3472, 4255, 1499, 3730, 3905, 5321, 
    3700, 2101, 1238, 524, 2411, 4440, 1555, 2317, 165, 978, 1740, 411, 5734, 2810, 6004, 3642, 3376, 1188, 662, 3155, 5056, 6260, 5123, 1986, 4152, 2951, 2075, 6201, 2810, 372, 3819, 5637, 4810, 3540, 3273, 
    96, 1041, 3114, 2946, 5238, 2000, 2345, 2491, 1233, 174, 2385, 1240, 1266, 6372, 2369, 428, 5803, 2989, 2373, 5298, 1908, 1779, 934, 1533, 5698, 2183, 5436, 2797, 2766, 2771, 3183, 1118, 1779, 5066, 95, 
    2817, 4124, 215, 2527, 3355, 4092, 3765, 3998, 1436, 4252, 2949, 2367, 3327, 3243, 4678, 4710, 3410, 369, 2137, 4414, 3707, 1731, 4443, 6373, 2726, 5333, 3894, 5508, 2929, 5494, 6000, 3506, 4507, 121, 6043, 
    5860, 259, 1335, 2092, 1400, 1300, 4532, 5206, 5603, 3431, 1685, 2582, 6129, 2689, 6166, 3532, 3796, 6252, 6278, 4240, 2660, 6150, 6103, 441, 1727, 6060, 4856, 5020, 6399, 1357, 3628, 3862, 3868, 221, 4265, 
    633, 5071, 4520, 813, 4213, 1897, 4186, 3532, 4231, 4031, 3233, 2523, 1512, 6376, 2753, 6132, 3906, 4, 2861, 2174, 3349, 182, 6051, 1350, 4286, 2877, 6091, 5322, 353, 5045, 3121, 970, 5751, 1990, 1435, 4958, 
    2214, 3819, 4769, 2738, 3610, 685, 3041, 4632, 830, 3259, 2710, 2259, 2636, 596, 6117, 4228, 2125, 1944, 578, 723, 192, 321, 5461, 5347, 1415, 218, 3874, 4014, 3428, 3498, 1582, 3564, 3968, 1539, 3458, 5557, 
    5419, 5072, 5442, 6193, 4065, 3992, 1180, 875, 4313, 75, 4617, 4039, 4585, 1341, 3642, 938, 610, 5373, 900, 2694, 6225, 933, 3620, 470, 818, 498, 3689, 3989, 3328, 1233, 5667, 2573, 4986, 4028, 1080, 852, 4876, 
    3462, 1448, 1986, 3513, 1929, 3495, 25, 2186, 3908]
    dataset = read_recurrent_dataset(olddata, sequence_length, filter_ids=indices, mode='exclude')
    validation_set = read_recurrent_dataset(olddata, sequence_length, filter_ids=indices, mode='include')


callbacks = [
# Write TensorBoard logs to `./logs` directory
tf.keras.callbacks.TensorBoard(log_dir=archive_path+name+"_"+modus),
tf.keras.callbacks.ModelCheckpoint(filepath=model_path+"checkpoints/"+name+modus+".ckpt",
                                    save_best_only=True,
                                    period=20)
]


model.fit(x=dataset, 
        epochs=150,
        steps_per_epoch=6384//32,
        validation_data=validation_set,
        validation_steps=638//32,
        callbacks = callbacks)

tf.keras.models.save_model(model,archive_path+name+"_"+modus+"/"+name+"_"+modus+".h5",overwrite=True)


print("Mission accomplished.")