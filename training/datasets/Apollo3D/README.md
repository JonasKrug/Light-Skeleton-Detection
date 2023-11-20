# Converting Apollo 3D car instances to LSD Keypoint annotations:
We can utilize the Apollo ... dataset for our keypoint predction task.
For this we first utilize the label conversion script from [OpenPifPaf](https://openpifpaf.github.io/plugins_apollocar3d.html) and then run our own conversion on top of that.

First download and extract the dataset by running 
```
./download_training_data.sh 
```

Then make sure that the conda environment with OpenPifPaf installed is activated and convert the annotations using a script from OpenPifPaf:
```
python3 -m openpifpaf.plugins.apollocar3d.apollo_to_coco --dir_data train --dir_out train
```

And finally convert the annotations to the LSD keypoints by running:
```
./convert_training_data_and_clean_up.sh
```

The dataset is now ready to use to train the keypoint and bounding box detection tasks.
However, it does not yet provide any labels for the keypoint states and only provides recordings from one frontal camera.
