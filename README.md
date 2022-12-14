# Neural Architecture Search for Skeleton-based Human Action Recognition
This is the repository for the thesis "Neural Architecture Search for Skeleton-based Human Action Recognition".
## Abstract
Human action recognition has many applications, e. g., in human-computer interaction,
surveillance, sign language recognition, and human behavior analysis. Over the past
decades, researchers mainly focused on RGB-based action recognition. Due to advances
in skeleton estimation using RGB and depth modalities, skeleton data is becoming a
common modality in action recognition. For example, the skeleton data can be recorded
with Kinect cameras. 3D skeleton data can effectively represent motion dynamics and,
unlike color images, are not sensitive to light and scene variations.
While many neural architectures have been discovered for image classification and
object recognition, there is little work on a suitable architecture for skeleton-based ac-
tion recognition. For example, the NAS-Bench-201 search space contains 15625 neural
architectures, with some suitable for image classification, achieving test accuracies of
up to 93.97%.
This work contributes further research on skeleton-based action recognition by in-
vestigating the neural architectures in the NAS-Bench-201 search space. Especially,
the topology of neural architectures that perform well or poorly in classifying skeleton-
based human action recognition data sets is investigated. All architectures are trained on
the different action recognition data sets UTD-MHAD and NTU RGB+D. The data sets
vary significantly in size and the number of included test data, participants, and actions.
Knowledge about the topology of architectures with good and bad classification results
is fundamental for developing new architectures. By knowing the worst performing ar-
chitectures, unsuitable architectures can be filtered, making the architecture search more
efficient.
The analysis shows that architectures with shortcut connections and convolutions
achieve particularly good results. The best architectures of both data sets use a combi-
nation of 1x1 and 3x3 convolutions in addition to a shortcut connection. For the smaller
data set UTD-MHAD, the convolutions are performed in parallel on the input, and for
the larger data set, NTU RGB+D sequentially. On average, the best architectures use
more 3x3 than 1x1 convolutions. The worst architectures are those that discard the input
or use only average pooling.

## Structure of the repository
* "explanations" contains files explaining how to use the files in this repository to repeat the training of the associated work.
  * In *How_to_get_started.md*, it is explained how to replicate the training of this work, from installing the prerequisites, to downloading and transforming the data sets, to all the calls to start the training process.
  * *linux_commands_on_server.txt* contains all the calls executed on a Linux server to train all the architectures of the NAS-Bench-201 search space on the server.
  * *How_to_add_a_new_dataset.md* explains how to use the "helper_code" folder to add new data sets and train them on the NAS-Bench-201 search space.
* In "analysis" are all files relevant to replicate the analysis of the work.
  * *read_trained_data.ipynb* is the file to create all figures and information used in tables for the evaluation of the thesis.
  * *preprocess_trained_data.py* is needed to export all relevant information of the preprocessed training results into a json file that can be used in *read_trained_data.ipynb*.
  * *get_resnet_and_skip_archs.ipynb* was used to find all Ids for ResNet-6 and ResNet-36 architectures (explanation in the thesis)
* The files in "changed_code" are needed to replicate the training of the thesis. More information in *explanations/How_to_get_started.md*.
* The files in "helper_code" are the files that were used to transform all data of the UTD-MHAD, NTU RGB+D, and NTU RGB+D 120 data sets (last one was not trained). More information in *explanations/*How_to_add_a_new_dataset.md** and *explanations/How_to_get_started.md*.


