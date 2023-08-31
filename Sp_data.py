import splitfolders

input_folder = 'archive/CompleteImages/All data (Compressed)'
splitfolders.ratio(input_folder, 'data_splited', ratio = (0.75, 0.25), seed=13, group_prefix=None)


