import tensorflow as tf

# Load your trained model
model_path = r"C:\Users\Jefferson\Desktop\PROJECTS\Pest_Detection\pest_detection_model.h5"

model = tf.keras.models.load_model(model_path)


print("✅ Model loaded successfully.")

# Attempt to extract class labels
if 'class_names' in model.__dict__:
    class_labels = model.class_names  # Some models store labels here
elif hasattr(model, 'categories'):
    class_labels = model.categories  # Check another common location
else:
    # If labels are not stored in the model, use a predefined list or file
    labels_file = "path_to_labels.txt"  # Update if you have a labels file
    try:
        with open(labels_file, "r") as f:
            class_labels = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        class_labels = ["Unknown"] * model.output_shape[-1]  # Default placeholder

# Print the retrieved labels
print("\n📌 Retrieved Class Labels:")
for i, label in enumerate(class_labels):
    print(f"{i}: {label}")

# Optionally return the labels as a list
class_labels
