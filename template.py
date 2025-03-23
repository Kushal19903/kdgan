import os
from pathlib import Path

# Define the directory structure
structure = {
    "kdgan/": {
        "app.py": None,
        "requirements.txt": None,
        "setup.sh": None,
        "train.sh": None,
        "evaluate.sh": None,
        "run.sh": None,
        "static/": {
            "css/": {
                "styles.css": None,
            },
            "js/": {
                "main.js": None,
            },
            "images/": {
                "placeholder.jpg": None,
            },
        },
        "templates/": {
            "index.html": None,
            "generate.html": None,
            "gallery.html": None,
            "about.html": None,
            "layout.html": None,
        },
        "models/": None,
        "output/": {
            "images/": None,
        },
        "code/": {
            "cfg/": {
                "bird_KDGAN.yml": None,
                "coco_KDGAN.yml": None,
                "eval_bird.yml": None,
                "eval_coco.yml": None,
            },
            "miscc/": {
                "bert_encoder.py": None,
                "config.py": None,
                "model.py": None,
                "utils.py": None,
            },
            "datasets.py": None,
            "main.py": None,
        },
        "data/": {
            "birds/": None,
            "coco/": None,
        },
        "eval/": {
            "FID/": {
                "fid_score.py": None,
            },
            "IS/": {
                "bird/": {
                    "inception_score.py": None,
                },
                "coco/": {
                    "inception_score.py": None,
                },
            },
        },
    },
}

def create_structure(base_path, structure):
    """
    Recursively create the directory and file structure.
    """
    for name, content in structure.items():
        path = base_path / name
        if content is None:  # It's a file
            path.touch()  # Create an empty file
        else:  # It's a directory
            path.mkdir(parents=True, exist_ok=True)
            create_structure(path, content)  # Recursively create its contents

if __name__ == "__main__":
    base_dir = Path.cwd()  # Current working directory
    create_structure(base_dir, structure)
    print("File structure created successfully!")