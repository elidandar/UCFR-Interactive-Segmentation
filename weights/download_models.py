import os
import sys
import urllib.request
import urllib.error

def download_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, (downloaded * 100) // total_size)
        bar_length = 40
        filled_length = int(bar_length * percent // 100)
        bar = '=' * filled_length + '-' * (bar_length - filled_length)
        # Convert sizes to MB
        downloaded_mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        sys.stdout.write(f"\r[{bar}] {percent}% ({downloaded_mb:.1f} MB / {total_mb:.1f} MB)")
        sys.stdout.flush()
    else:
        sys.stdout.write(f"\rDownloaded {downloaded / (1024 * 1024):.1f} MB")
        sys.stdout.flush()

def download_weights(url, dest_path):
    dest_dir = os.path.dirname(dest_path)
    os.makedirs(dest_dir, exist_ok=True)
    
    print(f"Downloading model weights from: {url}")
    print(f"Saving to: {dest_path}")
    
    try:
        # User-Agent header to avoid block from some CDNs
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        
        urllib.request.urlretrieve(url, dest_path, download_progress)
        print("\nDownload completed successfully!")
        return True
    except urllib.error.URLError as e:
        print(f"\nError downloading model weights: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False
    except KeyboardInterrupt:
        print("\nDownload cancelled by user.")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False

def main():
    url = "https://github.com/elidandar/UCFR-Interactive-Segmentation/releases/download/v0.1.0/sbd_vit_base_ufcr.pth"
    dest_dir = os.path.dirname(os.path.abspath(__file__))
    dest_path = os.path.join(dest_dir, "sbd_vit_base_ufcr.pth")
    
    if os.path.exists(dest_path):
        print(f"Model weight file already exists at {dest_path}. Skipping download.")
        return
        
    success = download_weights(url, dest_path)
    if not success:
        sys.exit(1)

if __name__ == '__main__':
    main()
