import os

train_dir = r'C:\Users\adeif\OneDrive\Year 4\Sem 2\Computer Vision\Assignment 2\chest_xray\train'
test_dir = r'C:\Users\adeif\OneDrive\Year 4\Sem 2\Computer Vision\Assignment 2\chest_xray\test'

def count_images(folder):
    counts = {}
    total = 0

    for class_name in os.listdir(folder):
        class_path = os.path.join(folder, class_name)
        if os.path.isdir(class_path):
            num_files = len([
                f for f in os.listdir(class_path)
                if os.path.isfile(os.path.join(class_path, f))
            ])
            counts[class_name] = num_files
            total += num_files

    return counts, total

train_counts, train_total = count_images(train_dir)
test_counts, test_total = count_images(test_dir)

print("Training set distribution:")
for cls, count in train_counts.items():
    print(f"{cls}: {count}")
print(f"Total train images: {train_total}\n")

print("Test set distribution:")
for cls, count in test_counts.items():
    print(f"{cls}: {count}")
print(f"Total test images: {test_total}")