import gdown
import sys
import os
import zipfile


def download_network_weights():
    url = 'https://drive.google.com/file/d/1dbN7LdMV03BR1Eda8n7iKNIyYp9r07sM'
    output = os.path.join(os.path.dirname(__file__),
                          'resources/soccrnet_weights.zip')
    try:
        gdown.download(url, output, quiet=False)
        print(f"Network weights downloaded successfully and saved to {output}")
        with zipfile.ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall(os.path.join(
                os.path.dirname(__file__), 'resources'))
        print("Network weights unzipped successfully")
    except Exception as e:
        print(f"An error occurred while downloading the data: {e}")
        sys.exit(1)


if __name__ == "__main__":
    download_network_weights()
