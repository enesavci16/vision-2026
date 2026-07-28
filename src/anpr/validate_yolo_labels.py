from pathlib import Path



input_path = Path(
    r"C:\Users\enesa\projeler\lead-traffic-projects\project-01\vision_2026\datasets\openalpr_benchmarks\endtoend"
)
datasets = ["br", "eu", "us"]


def parse_line(line: str):
    filename, x, y, w, h, plate_text = line.strip().split()
    return filename, x


for dataset in datasets:
    dataset_path = input_path / dataset

    for filename_path in dataset_path.glob("*.txt"):
        target_txt_path = filename_path.with_suffix(".txt")
        if target_txt_path.exists():
            with open (target_txt_path,"r") as file:
                for line in file:
                    _,x=parse_line(line)
                    print(x)
                


