import json
from pathlib import Path

#change the head folder
DATASET_HEAD_FOLDER = Path("datasets/global_daily_object_yolo/Dollar street trial")


classid = {}
objects = 0

with open(DATASET_HEAD_FOLDER / "ann_to_yolo_dataset_converter.log", "w") as f:
    for folder in DATASET_HEAD_FOLDER.iterdir():
        if not folder.is_dir():
            continue

        OUTPUT_DIR = folder / "ann_yolo"
        OUTPUT_DIR.mkdir(exist_ok=True)

        print(f"opening: {folder}")
        for json_file in folder.rglob("*.json"):
            f.write(str(json_file)+"\n")

            with open(json_file, "r") as f_json:
                data = json.load(f_json)

            image_width = data["size"]["width"]
            image_height = data["size"]["height"]

            yolo_lines = []

            for obj in data["objects"]:
                class_name = obj["classTitle"]
                class_id = obj["classId"]
                if class_name not in classid:
                    classid[class_name] = [class_id, objects]
                    objects+=1
                f.write(class_name + "_" + str(class_id) + ", ")

                #get bounding box
                points = obj["points"]["exterior"]

                x1, y1 = points[0]
                x2, y2 = points[1]

                #convert to yolo format
                x_center = (x1 + x2)/2
                y_center = (y1 + y2)/2

                width = x2 - x1
                height = y2 - y1

                #normalize
                x_center /= image_width
                y_center /= image_height
                width /= image_width
                height /= image_height

                yolo_lines.append(
                    f"{classid[class_name][1]} "
                    f"{x_center:.6f} "
                    f"{y_center:.6f} "
                    f"{width:.6f} "
                    f"{height:.6f}"
                )

            #save YOLO annotation 
            output_file = OUTPUT_DIR / (json_file.name.removesuffix(".jpg.json") + ".txt")

            with open(output_file, "w") as f_output:
                f_output.write("\n".join(yolo_lines))


with open(DATASET_HEAD_FOLDER / "meta_yolo.yaml", "w") as f:
    lines = f"""
path: {DATASET_HEAD_FOLDER}
train: images/train
val: images/val
names:
"""
    f.write(lines)
    print(classid)
    for class_name, id in classid.items():
        f.write("  "+ str(id[1]) + ": "+ class_name + "\n")

    f.write("\noriginal_ids:\n")
    for class_name, id in classid.items():
            f.write("  "+ str(id[0]) + ": "+ class_name + "\n")


    


