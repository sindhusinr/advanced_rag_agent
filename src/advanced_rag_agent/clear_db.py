from pathlib import Path
import shutil

db_path = Path("chroma_db")

if db_path.exists():

    for item in db_path.iterdir():

        if item.is_dir():
            shutil.rmtree(item)

        else:
            item.unlink()

    print("Chroma DB contents cleared ✅")

else:
    print("Chroma DB does not exist.")