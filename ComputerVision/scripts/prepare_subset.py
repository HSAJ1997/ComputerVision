import os
import csv
import json
import random
import tarfile
import shutil
from collections import defaultdict


random.seed(42)

classes = 500
train_amount = 40
valid_amount = 10
test_amount = 10

folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

train_json = os.path.join(folder, "data", "train_mini.json")
test_json = os.path.join(folder, "data", "val.json")

train_tar = os.path.join(folder, "train_mini.tar.gz")
test_tar = os.path.join(folder, "val.tar.gz")

split_folder = os.path.join(folder, "splits")
image_folder = os.path.join(folder, "subset")

os.makedirs(split_folder, exist_ok=True)
os.makedirs(image_folder, exist_ok=True)

with open(train_json, "r", encoding="utf-8") as file:
    train_data = json.load(file)

with open(test_json, "r", encoding="utf-8") as file:
    test_data = json.load(file)

train_images = {}

for image in train_data["images"]:
    train_images[image["id"]] = image

test_images = {}

for image in test_data["images"]:
    test_images[image["id"]] = image

train_groups = defaultdict(list)

for item in train_data["annotations"]:
    image_id = item["image_id"]
    class_id = item["category_id"]

    train_groups[class_id].append(train_images[image_id])

test_groups = defaultdict(list)

for item in test_data["annotations"]:
    image_id = item["image_id"]
    class_id = item["category_id"]

    test_groups[class_id].append(test_images[image_id])

names = {}

for item in train_data["categories"]:
    names[item["id"]] = item.get("name", "")

available = []

for class_id in train_groups:
    if len(train_groups[class_id]) >= 50 and len(test_groups[class_id]) >= 10:
        available.append(class_id)

chosen = random.sample(sorted(available), classes)
chosen.sort()

train_rows = []
valid_rows = []
test_rows = []

class_list = []

train_wanted = {}
test_wanted = {}


def fix_path(path):
    return path.replace("\\", "/").lstrip("./")


def short_path(path):
    path = fix_path(path)
    parts = path.split("/")

    if len(parts) > 1:
        return "/".join(parts[1:])

    return path


for number, class_id in enumerate(chosen):
    pictures = train_groups[class_id].copy()
    random.shuffle(pictures)

    train_pictures = pictures[:train_amount]
    valid_pictures = pictures[
        train_amount:
        train_amount + valid_amount
    ]

    test_pictures = test_groups[class_id][:test_amount]

    species = names.get(class_id, "")

    class_list.append({
        "class_index": number,
        "category_id": class_id,
        "species_name": species
    })

    for image in train_pictures:
        old_path = fix_path(image["file_name"])
        new_path = short_path(old_path)

        train_rows.append({
            "image_path": (
                "subset/train_mini/" + new_path
            ),
            "category_id": class_id,
            "class_index": number,
            "species_name": species,
            "split": "train"
        })

        train_wanted[old_path] = new_path
        train_wanted[new_path] = new_path

    for image in valid_pictures:
        old_path = fix_path(image["file_name"])
        new_path = short_path(old_path)

        valid_rows.append({
            "image_path": (
                "subset/train_mini/" + new_path
            ),
            "category_id": class_id,
            "class_index": number,
            "species_name": species,
            "split": "validation"
        })

        train_wanted[old_path] = new_path
        train_wanted[new_path] = new_path

    for image in test_pictures:
        old_path = fix_path(image["file_name"])
        new_path = short_path(old_path)

        test_rows.append({
            "image_path": (
                "subset/val/" + new_path
            ),
            "category_id": class_id,
            "class_index": number,
            "species_name": species,
            "split": "test"
        })

        test_wanted[old_path] = new_path
        test_wanted[new_path] = new_path


headings = [
    "image_path",
    "category_id",
    "class_index",
    "species_name",
    "split"
]


def make_csv(name, rows):
    path = os.path.join(split_folder, name)

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=headings)
        writer.writeheader()
        writer.writerows(rows)


make_csv("train.csv", train_rows)
make_csv("validation.csv", valid_rows)
make_csv("test.csv", test_rows)

with open(
    os.path.join(split_folder, "selected_classes.json"),
    "w",
    encoding="utf-8"
) as file:
    json.dump({
        "seed": 42,
        "number_of_classes": classes,
        "train_per_class": train_amount,
        "validation_per_class": valid_amount,
        "test_per_class": test_amount,
        "classes": class_list
    }, file, indent=2)


def extract_images(tar_path, wanted, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    needed = set(wanted.values())
    done = set()

    with tarfile.open(tar_path, "r:gz") as archive:
        for item in archive:
            if not item.isfile():
                continue

            full_name = fix_path(item.name)
            small_name = short_path(full_name)

            new_name = None

            if full_name in wanted:
                new_name = wanted[full_name]
            elif small_name in wanted:
                new_name = wanted[small_name]

            if new_name is None or new_name in done:
                continue

            save_path = os.path.join(
                output_folder,
                *new_name.split("/")
            )

            os.makedirs(
                os.path.dirname(save_path),
                exist_ok=True
            )

            image = archive.extractfile(item)

            if image is not None:
                with image:
                    with open(save_path, "wb") as output:
                        shutil.copyfileobj(image, output)

                done.add(new_name)

            if len(done) == len(needed):
                break

    missing = needed - done

    if len(missing) > 0:
        print("Missing images:", len(missing))
        print(list(missing)[:5])
    else:
        print("All images extracted")


extract_images(
    train_tar,
    train_wanted,
    os.path.join(image_folder, "train_mini")
)

extract_images(
    test_tar,
    test_wanted,
    os.path.join(image_folder, "val")
)


print("FINISHED :3")