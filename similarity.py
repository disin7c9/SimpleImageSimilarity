import os
import numpy as np
import cv2 as cv
from PIL import Image, ImageTk
from imagehash import phash
import tkinter as tk
from tkinter import filedialog, messagebox


def similarity_cv(input_path):
    input_image = cv.imread(input_path, cv.IMREAD_GRAYSCALE)
    image_files = [f for f in os.listdir("./imageset") if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    images = [cv.imread(os.path.join("./imageset", image_file), cv.IMREAD_GRAYSCALE) for image_file in image_files]

    orb = cv.ORB_create()
    kp1, des1 = orb.detectAndCompute(input_image, None)
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    similarity_scores = []
    for idx, img2 in enumerate(images):
        kp2, des2 = orb.detectAndCompute(img2, None)
        if kp1 is None or des1 is None or kp2 is None or des2 is None:
            similarity_scores.append((0, idx))
            continue

        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        score = len(matches)
        percentage = (score / len(kp1)) * 100 if kp1 else 0
        similarity_scores.append((percentage, idx))

    return similarity_scores


def similarity_phash(input_path):
    input_image = Image.open(input_path).convert('L')
    image_files = [f for f in os.listdir("./imageset") if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    images = [Image.open(os.path.join("./imageset", image_file)).convert('L') for image_file in image_files]

    hash1 = phash(input_image)
    similarity_scores = [((1 - (hash1 - phash(img)) / len(hash1.hash.flatten())) * 100, idx) for idx, img in enumerate(images)]
    
    return similarity_scores


class ImageSimilarityApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Similarity App")

        self.header_label = tk.Label(root, text="Asterinnet - Image similarity", font=("Arial", 16))
        self.header_label.pack(pady=10)
        
        self.input_path = None
        self.label = tk.Label(root, text="Select an input image to start")
        self.label.pack()

        button_frame = tk.Frame(root)
        button_frame.pack()

        self.select_button = tk.Button(button_frame, text="Select Image", command=self.select_image)
        self.select_button.pack(side=tk.LEFT)

        self.reset_button = tk.Button(button_frame, text="Reset", command=self.reset)
        self.reset_button.pack(side=tk.LEFT)

        self.result_label = tk.Label(root, text="")
        self.result_label.pack()

        self.image_frame = tk.Frame(root)
        self.image_frame.pack()

    def select_image(self):
        self.reset()
        self.input_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if self.input_path:
            similarity_cv_scores = similarity_cv(self.input_path)
            similarity_phash_scores = similarity_phash(self.input_path)

            weighted_scores = []
            for cv_score, phash_score in zip(similarity_cv_scores, similarity_phash_scores):
                avg_score = (cv_score[0] + phash_score[0]) / 2
                weighted_scores.append((avg_score, cv_score[1]))

            weighted_scores.sort(reverse=True, key=lambda x: x[0])
            best_match = weighted_scores[0]
            best_match_idx = best_match[1]

            image_files = [f for f in os.listdir("./imageset") if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            best_match_image_path = os.path.join("./imageset", image_files[best_match_idx])

            self.display_result(best_match_image_path, best_match[0], similarity_cv_scores[best_match_idx][0], similarity_phash_scores[best_match_idx][0])

    def display_result(self, image_path, weighted_score, cv_score, phash_score):
        input_image = Image.open(self.input_path)
        best_match_image = Image.open(image_path)

        input_image = self.resize_image(input_image, (800, 450))
        best_match_image = self.resize_image(best_match_image, (800, 450))

        input_image = ImageTk.PhotoImage(input_image)
        best_match_image = ImageTk.PhotoImage(best_match_image)

        self.result_label.config(text=f"Weighted Similarity: {weighted_score:.2f}% (ORB: {cv_score:.2f}%, pHash: {phash_score:.2f}%)")

        input_frame = tk.LabelFrame(self.image_frame, text="Input Image", padx=10, pady=10, bd=2, relief="solid")
        input_frame.grid(row=0, column=0, padx=10, pady=10)
        input_image_label = tk.Label(input_frame, image=input_image)
        input_image_label.pack()
        input_image_label.image = input_image

        match_frame = tk.LabelFrame(self.image_frame, text="Most Similar Image", padx=10, pady=10, bd=2, relief="solid")
        match_frame.grid(row=0, column=1, padx=10, pady=10)
        best_match_image_label = tk.Label(match_frame, image=best_match_image)
        best_match_image_label.pack()
        best_match_image_label.image = best_match_image

        messagebox.showinfo("Result", f"Most similar image found with {weighted_score:.2f}% similarity (ORB: {cv_score:.2f}%, pHash: {phash_score:.2f}%)")

    def resize_image(self, image, max_size):
        image.thumbnail(max_size, Image.LANCZOS)
        return image

    def reset(self):
        for widget in self.image_frame.winfo_children():
            widget.destroy()
        self.result_label.config(text="")
        self.label.config(text="Select an input image to start")
        self.input_path = None


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSimilarityApp(root)
    root.mainloop()
