# How to add a new data set
1. Transform the 3D skeletal data of the new data set to RGB images.
For example, you can adapt the file *code/NTU_60_transform_skeleton_data_to_images.py*.
2. Calculate mean and standard deviation for the training data
For example, you can adapt the file *code/ntu_60_get_statistical_data.py*.
3. Create a class to load the images of your new data set.
   1. You should inherit from class [VISIONDATASET](https://pytorch.org/vision/stable/generated/torchvision.datasets.VisionDataset.html).
   For example, you can adapt the file *code/NTU_60_Dataset.py* or for example the [cifar10](https://pytorch.org/vision/stable/_modules/torchvision/datasets/cifar.html#CIFAR10) data set from torchvision.datasets.
   2. If not all data points have the same size/shape, like the data sets used in this thesis, you can use functions from the file *changed_code/xautodl/datasets/ntu_60_get_and_transform_dataset.py* to transform the data.
   3. Move the file that includes the class to the directory *changed_code/xautodl/datasets/* and add the class to the file *\_\_init\_\_.py* in this directory.
4. Extend *changed_code/xautodl/datasets/own_get_dataset_with_transform_no_augmentation.py* to load your data set.
   1. Add the number of classes to the dict <b>Dataset2Class</b>.
   2. Extend the function <b>get_datasets_no_augmentation</b> (or without "_no_augmentation") to load the new data set.
      1. Add your desired data set transformations in the section #Data Augmentation. In this thesis only the operations transforms.ToTensor(),
            and transforms.Normalize(mean, std) were used.
      2. Write the shape of your data set in the xshape variable. xshape = (1, 3, height_image, width_image). height_image and width_image should be multiples of 4.
      3. Load the train and test data in the variables train_data and test_data. You should use the class created in 3. for this.
   3. Hint: You can search the code for every occurrence of 'utdmhad' or 'ntu' and adapt the code to the new data set.
   4. If you create a new file in the directory *changed_code/xautodl/datasets* adapt the file *\_\_init\_\_.py* in this directory.
5. Extend the file *changed_code/exps/NATS-Bench/own_main-tss_no_augmentation.py* (or without "_no_augmentation") .
   1. Call the new version of the function <b>get_datasets_no_augmentation</b> that you extended (not needed if you do not create new files).
   2. Split the train data into a train and a validation split. Create a train and validation loader with the corresponding data.
   3. Hint: You can search the code for every occurrence of 'utdmhad' or 'ntu' and adapt the code to the new data set.
6. Call the changed *changed_code/exps/NATS-Bench/own_main-tss_no_augmentation.py*.
   1. Hint: Copy a file of the directory *changed_code/scripts/NATS-Bench* and adapt it to the new data set.
7. Copy all files from changed_code to the corresponding directory in the XAutoDL repository and follow the instructions from *explanations/How_to_get_started.md* from "<b>Install changed XAutoDL repository</b>"