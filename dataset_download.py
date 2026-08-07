import kagglehub

# Download directly to your custom folder
path = kagglehub.dataset_download("msambare/fer2013", output_dir="./dataset")

print("Path to dataset files:", path)