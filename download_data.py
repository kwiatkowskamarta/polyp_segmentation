import os
import requests
import zipfile
from tqdm import tqdm
import urllib3

# Suppress the warning that pops up when we disable SSL verification
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DATA_URL = "https://datasets.simula.no/downloads/kvasir-seg.zip"
ZIP_PATH = "data/kvasir-seg.zip"
EXTRACT_PATH = "data/"

def download_data():
    if not os.path.exists("data"):
        os.makedirs("data")

    if os.path.exists(os.path.join(EXTRACT_PATH, "Kvasir-SEG")):
        print("Data already exists.")
        return

    print("Downloading Kvasir-SEG dataset...")
    
    #verify=False ignores the SSL certificate error
    response = requests.get(DATA_URL, stream=True, verify=False)
    
    total_size = int(response.headers.get('content-length', 0))

    with open(ZIP_PATH, "wb") as file, tqdm(
        desc="Progress",
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

    print("Unzipping...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_PATH)
    
    print("Done! Data located in data/Kvasir-SEG")

if __name__ == "__main__":
    download_data()