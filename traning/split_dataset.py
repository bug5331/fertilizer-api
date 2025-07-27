import splitfolders

input_folder = "crop_disease_dataset"
output_folder = "dataset_split"

splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(.7, .15, .15))

print("✅ Done")
