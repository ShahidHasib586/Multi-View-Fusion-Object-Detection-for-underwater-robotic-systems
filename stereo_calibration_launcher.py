#!/usr/bin/env python3
import os
import sys
import shlex
import subprocess
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


DEFAULT_SCRIPT = "/home/shahid/multicam_3d/scripts/stereo_calibrate_from_videos.py"


class StereoCalibrationLauncher(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Stereo Calibration Launcher")
        self.geometry("900x700")

        self.proc = None

        self.script_path = tk.StringVar(value=DEFAULT_SCRIPT)
        self.left_video = tk.StringVar(value="/home/shahid/multicam_3d/data/left_calib.mp4")
        self.right_video = tk.StringVar(value="/home/shahid/multicam_3d/data/right_calib.mp4")
        self.board_cols = tk.StringVar(value="9")
        self.board_rows = tk.StringVar(value="5")
        self.square_size = tk.StringVar(value="0.025")
        self.frame_step = tk.StringVar(value="5")
        self.max_pairs = tk.StringVar(value="50")
        self.out_dir = tk.StringVar(value="calib_out")
        self.show_flag = tk.BooleanVar(value=True)
        self.python_exec = tk.StringVar(value=sys.executable)

        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 8, "pady": 6}

        top = ttk.Frame(self)
        top.pack(fill="x", padx=10, pady=10)

        ttk.Label(top, text="Python:").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(top, textvariable=self.python_exec, width=70).grid(row=0, column=1, sticky="ew", **pad)
        ttk.Button(top, text="Browse", command=self.browse_python).grid(row=0, column=2, **pad)

        ttk.Label(top, text="Script:").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(top, textvariable=self.script_path, width=70).grid(row=1, column=1, sticky="ew", **pad)
        ttk.Button(top, text="Browse", command=self.browse_script).grid(row=1, column=2, **pad)

        ttk.Label(top, text="Left video:").grid(row=2, column=0, sticky="w", **pad)
        ttk.Entry(top, textvariable=self.left_video, width=70).grid(row=2, column=1, sticky="ew", **pad)
        ttk.Button(top, text="Browse", command=lambda: self.browse_file(self.left_video)).grid(row=2, column=2, **pad)

        ttk.Label(top, text="Right video:").grid(row=3, column=0, sticky="w", **pad)
        ttk.Entry(top, textvariable=self.right_video, width=70).grid(row=3, column=1, sticky="ew", **pad)
        ttk.Button(top, text="Browse", command=lambda: self.browse_file(self.right_video)).grid(row=3, column=2, **pad)

        opts = ttk.LabelFrame(self, text="Calibration Settings")
        opts.pack(fill="x", padx=10, pady=8)

        ttk.Label(opts, text="Board cols:").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(opts, textvariable=self.board_cols, width=10).grid(row=0, column=1, sticky="w", **pad)

        ttk.Label(opts, text="Board rows:").grid(row=0, column=2, sticky="w", **pad)
        ttk.Entry(opts, textvariable=self.board_rows, width=10).grid(row=0, column=3, sticky="w", **pad)

        ttk.Label(opts, text="Square size:").grid(row=0, column=4, sticky="w", **pad)
        ttk.Entry(opts, textvariable=self.square_size, width=10).grid(row=0, column=5, sticky="w", **pad)

        ttk.Label(opts, text="Frame step:").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(opts, textvariable=self.frame_step, width=10).grid(row=1, column=1, sticky="w", **pad)

        ttk.Label(opts, text="Max pairs:").grid(row=1, column=2, sticky="w", **pad)
        ttk.Entry(opts, textvariable=self.max_pairs, width=10).grid(row=1, column=3, sticky="w", **pad)

        ttk.Label(opts, text="Output dir:").grid(row=1, column=4, sticky="w", **pad)
        ttk.Entry(opts, textvariable=self.out_dir, width=18).grid(row=1, column=5, sticky="w", **pad)

        ttk.Checkbutton(opts, text="Show detections (--show)", variable=self.show_flag).grid(
            row=2, column=0, columnspan=3, sticky="w", **pad
        )

        cmd_box = ttk.LabelFrame(self, text="Command Preview")
        cmd_box.pack(fill="both", padx=10, pady=8)

        self.cmd_text = tk.Text(cmd_box, height=5, wrap="word")
        self.cmd_text.pack(fill="both", expand=True, padx=8, pady=8)
        self.update_command_preview()

        btns = ttk.Frame(self)
        btns.pack(fill="x", padx=10, pady=8)

        ttk.Button(btns, text="Update Preview", command=self.update_command_preview).pack(side="left", padx=5)
        ttk.Button(btns, text="Run Calibration", command=self.run_calibration).pack(side="left", padx=5)
        ttk.Button(btns, text="Stop", command=self.stop_calibration).pack(side="left", padx=5)

        log_box = ttk.LabelFrame(self, text="Output Log")
        log_box.pack(fill="both", expand=True, padx=10, pady=8)

        self.log_text = tk.Text(log_box, wrap="word")
        self.log_text.pack(fill="both", expand=True, padx=8, pady=8)

        top.columnconfigure(1, weight=1)

        for var in [
            self.python_exec, self.script_path, self.left_video, self.right_video,
            self.board_cols, self.board_rows, self.square_size,
            self.frame_step, self.max_pairs, self.out_dir
        ]:
            var.trace_add("write", lambda *args: self.update_command_preview())
        self.show_flag.trace_add("write", lambda *args: self.update_command_preview())

    def browse_python(self):
        path = filedialog.askopenfilename(title="Select Python executable")
        if path:
            self.python_exec.set(path)

    def browse_script(self):
        path = filedialog.askopenfilename(
            title="Select calibration script",
            filetypes=[("Python files", "*.py"), ("All files", "*.*")]
        )
        if path:
            self.script_path.set(path)

    def browse_file(self, target_var):
        path = filedialog.askopenfilename(
            title="Select video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if path:
            target_var.set(path)

    def build_command_list(self):
        cmd = [
            self.python_exec.get().strip(),
            self.script_path.get().strip(),
            "--left_video", self.left_video.get().strip(),
            "--right_video", self.right_video.get().strip(),
            "--board_cols", self.board_cols.get().strip(),
            "--board_rows", self.board_rows.get().strip(),
            "--square_size", self.square_size.get().strip(),
            "--frame_step", self.frame_step.get().strip(),
            "--max_pairs", self.max_pairs.get().strip(),
            "--out_dir", self.out_dir.get().strip(),
        ]
        if self.show_flag.get():
            cmd.append("--show")
        return cmd

    def update_command_preview(self):
        cmd = self.build_command_list()
        preview = " ".join(shlex.quote(x) for x in cmd)
        self.cmd_text.delete("1.0", tk.END)
        self.cmd_text.insert(tk.END, preview)

    def validate_inputs(self):
        if not os.path.isfile(self.python_exec.get().strip()):
            messagebox.showerror("Error", "Python executable not found.")
            return False
        if not os.path.isfile(self.script_path.get().strip()):
            messagebox.showerror("Error", "Calibration script not found.")
            return False
        if not os.path.isfile(self.left_video.get().strip()):
            messagebox.showerror("Error", "Left video not found.")
            return False
        if not os.path.isfile(self.right_video.get().strip()):
            messagebox.showerror("Error", "Right video not found.")
            return False

        try:
            int(self.board_cols.get().strip())
            int(self.board_rows.get().strip())
            float(self.square_size.get().strip())
            int(self.frame_step.get().strip())
            int(self.max_pairs.get().strip())
        except ValueError:
            messagebox.showerror("Error", "Numeric settings are invalid.")
            return False

        return True

    def append_log(self, text):
        self.log_text.insert(tk.END, text)
        self.log_text.see(tk.END)

    def run_calibration(self):
        if self.proc is not None:
            messagebox.showwarning("Busy", "Calibration is already running.")
            return

        if not self.validate_inputs():
            return

        cmd = self.build_command_list()
        self.append_log("\n=== Running ===\n")
        self.append_log(" ".join(shlex.quote(x) for x in cmd) + "\n\n")

        def worker():
            try:
                self.proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )
                for line in self.proc.stdout:
                    self.log_text.after(0, self.append_log, line)
                rc = self.proc.wait()
                self.log_text.after(0, self.append_log, f"\n=== Process finished with exit code {rc} ===\n")
            except Exception as e:
                self.log_text.after(0, self.append_log, f"\n[ERROR] {e}\n")
            finally:
                self.proc = None

        threading.Thread(target=worker, daemon=True).start()

    def stop_calibration(self):
        if self.proc is None:
            self.append_log("\n[INFO] No running process.\n")
            return
        try:
            self.proc.terminate()
            self.append_log("\n[INFO] Stop signal sent.\n")
        except Exception as e:
            self.append_log(f"\n[ERROR] Could not stop process: {e}\n")


if __name__ == "__main__":
    app = StereoCalibrationLauncher()
    app.mainloop()
