import matplotlib
matplotlib.use('Agg')

import argparse
import tkinter as tk
from tkinter import messagebox, ttk

import torch

from isegm.utils import exp
from isegm.inference import utils
from interactive_demo.app import InteractiveDemoApp

import sys
import os
import urllib.request
import urllib.error
import threading

class TkinterModelDownloader:
    def __init__(self, url, dest_path):
        self.url = url
        self.dest_path = dest_path
        self.success = False
        self.is_cancelled = False

        self.root = tk.Tk()
        self.root.title("Downloading Model Weights")
        self.root.geometry("400x150")
        self.root.resizable(False, False)

        # Center window
        self.root.eval('tk::PlaceWindow . center')

        # Layout
        self.lbl_status = ttk.Label(self.root, text="Downloading model weights...", font=("Helvetica", 11, "bold"))
        self.lbl_status.pack(pady=10)

        self.progress = ttk.Progressbar(self.root, orient="horizontal", length=300, mode="determinate")
        self.progress.pack(pady=10)

        self.lbl_info = ttk.Label(self.root, text="0.0 MB / -- MB (0.0 MB/s)")
        self.lbl_info.pack(pady=2)

        self.btn_cancel = ttk.Button(self.root, text="Cancel", command=self.cancel)
        self.btn_cancel.pack(pady=5)

        self.root.protocol("WM_DELETE_WINDOW", self.cancel)

    def start(self):
        self.thread = threading.Thread(target=self.run)
        self.thread.start()
        self.root.mainloop()
        return self.success

    def cancel(self):
        if messagebox.askyesno("Cancel Download", "Are you sure you want to cancel the model download?"):
            self.is_cancelled = True
            self.root.destroy()

    def run(self):
        try:
            import time
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
            urllib.request.install_opener(opener)

            response = urllib.request.urlopen(self.url)
            total_size = int(response.headers.get('content-length', 0))
            
            dest_dir = os.path.dirname(self.dest_path)
            if dest_dir:
                os.makedirs(dest_dir, exist_ok=True)

            downloaded = 0
            start_time = time.time()
            chunk_size = 1024 * 64

            with open(self.dest_path, 'wb') as f:
                while not self.is_cancelled:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    elapsed = time.time() - start_time
                    speed = (downloaded / elapsed) if elapsed > 0 else 0
                    percent = int(downloaded * 100 / total_size) if total_size > 0 else 0
                    
                    self.root.after(0, self.update_progress, percent, downloaded, total_size, speed)

            if self.is_cancelled:
                if os.path.exists(self.dest_path):
                    try:
                        os.remove(self.dest_path)
                    except OSError:
                        pass
            else:
                self.success = True
                self.root.after(0, self.finish)

        except Exception as e:
            if os.path.exists(self.dest_path):
                try:
                    os.remove(self.dest_path)
                except OSError:
                    pass
            self.root.after(0, self.show_error, str(e))

    def update_progress(self, percent, downloaded, total, speed):
        if self.is_cancelled:
            return
        self.progress['value'] = percent
        dl_mb = downloaded / (1024 * 1024)
        tot_mb = total / (1024 * 1024)
        sp_mb = speed / (1024 * 1024)
        self.lbl_info.config(text=f"{dl_mb:.1f} MB / {tot_mb:.1f} MB ({sp_mb:.1f} MB/s)")

    def finish(self):
        messagebox.showinfo("Download Complete", "Model weights downloaded successfully!")
        self.root.destroy()

    def show_error(self, err_msg):
        messagebox.showerror("Download Error", f"An error occurred while downloading weights:\n{err_msg}")
        self.root.destroy()

def main():
    args, cfg = parse_args()

    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        resolved_path = os.path.join(cfg.INTERACTIVE_MODELS_PATH, os.path.basename(checkpoint_path))
        if os.path.exists(resolved_path):
            checkpoint_path = resolved_path
        else:
            # Checkpoint file is missing! Show Tkinter model downloader.
            url = "https://github.com/elidandar/UCFR-Interactive-Segmentation/releases/download/v0.1.0/sbd_vit_base_ufcr.pth"
            dest_path = resolved_path if resolved_path.endswith('.pth') else os.path.join(cfg.INTERACTIVE_MODELS_PATH, "sbd_vit_base_ufcr.pth")
            
            downloader = TkinterModelDownloader(url, dest_path)
            if not downloader.start():
                print("Download cancelled or failed. Exiting.")
                sys.exit(0)
            checkpoint_path = dest_path

    torch.backends.cudnn.deterministic = True
    resolved_checkpoint = utils.find_checkpoint(cfg.INTERACTIVE_MODELS_PATH, checkpoint_path)
    model = utils.load_is_model(resolved_checkpoint, args.device, args.eval_ritm, cpu_dist_maps=True)

    root = tk.Tk()
    root.minsize(960, 480)
    app = InteractiveDemoApp(root, args, model)
    root.deiconify()
    app.mainloop()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default = './weights/sbd_vit_base_ufcr.pth',
                        required=False,
                        help='The path to the checkpoint. '
                             'This can be a relative path (relative to cfg.INTERACTIVE_MODELS_PATH) '
                             'or an absolute path. The file extension can be omitted.')

    parser.add_argument('--gpu', type=int, default=0,
                        help='Id of GPU to use.')

    parser.add_argument('--cpu', action='store_true', default=True,
                        help='Use only CPU for inference.')

    parser.add_argument('--limit-longest-size', type=int, default=800,
                        help='If the largest side of an image exceeds this value, '
                             'it is resized so that its largest side is equal to this value.')

    parser.add_argument('--cfg', type=str, default="config.yml",
                        help='The path to the config file.')

    parser.add_argument('--eval-ritm', action='store_true', default=False)

    args = parser.parse_args()
    if args.cpu:
        args.device =torch.device('cpu')
    else:
        args.device = torch.device(f'cuda:{args.gpu}')
    cfg = exp.load_config_file(args.cfg, return_edict=True)

    return args, cfg


if __name__ == '__main__':
    main()
