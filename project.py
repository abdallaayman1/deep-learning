import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageEnhance

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder


class PotatoLeafDeepLearner:
    # ================= fakhry =================
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Potato Leaf")
        self.root.geometry("1000x900")
        self.root.configure(bg="#121212")

        self.model = None
        self.dataset_path = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_map = {}
        self.CONFIDENCE_THRESHOLD = 0.60

        
        tk.Label(
            root,
            text="Potato Leaf Disease (Deep Learning)",
            font=("Arial", 22, "bold"),
            bg="#121212",
            fg="#FFFFFF"
        ).pack(pady=20)

        btn_frame = tk.Frame(root, bg="#121212")
        btn_frame.pack(pady=10)

        ttk.Button(btn_frame, text="1. Load Dataset", command=self.load_dataset)\
            .grid(row=0, column=0, padx=10)

        self.train_btn = ttk.Button(btn_frame, text="2. Train Model", command=self.start_training)
        self.train_btn.grid(row=0, column=1, padx=10)

        self.detect_btn = ttk.Button(btn_frame, text="3. Detect Image", command=self.detect_disease)
        self.detect_btn.grid(row=0, column=2, padx=10)

        self.status_label = tk.Label(
            root, text="Status: Ready",
            font=("Arial", 12), bg="#121212", fg="#03DAC6"
        )
        self.status_label.pack(pady=5)

        self.acc_label = tk.Label(
            root, text="",
            font=("Arial", 14, "bold"), bg="#121212", fg="#00FF00"
        )
        self.acc_label.pack(pady=5)

        self.result_label = tk.Label(
            root, text="",
            font=("Arial", 18, "bold"), bg="#121212"
        )
        self.result_label.pack(pady=10)

        self.image_label = tk.Label(
            root, bg="#1e1e1e", bd=2, relief="sunken"
        )
        self.image_label.pack(pady=10, padx=20, fill="both", expand=True)

        # Load saved model if exists
        if os.path.exists("potato_deep_model.pth"):
            self.load_saved_model()

    # ================= abdalla =================
    def load_dataset(self):
        self.dataset_path = filedialog.askdirectory(title="Select Dataset Folder")
        if self.dataset_path:
            self.status_label.config(text="Dataset Loaded Successfully")
            self.acc_label.config(text="")

    def start_training(self):
        self.train_btn.config(state="disabled")
        threading.Thread(target=self.train_model, daemon=True).start()

    def train_model(self):
        if not self.dataset_path:
            messagebox.showerror("Error", "Load dataset first!")
            self.train_btn.config(state="normal")
            return

        self.status_label.config(text="Preparing data & model...")

        # -------- Data Augmentation --------
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

        full_dataset = ImageFolder(self.dataset_path, transform=train_transform)
        class_names = sorted(full_dataset.class_to_idx.keys())
        num_classes = len(class_names)

        self.label_map = {
            i: name.replace("_", " ").title()
            for i, name in enumerate(class_names)
        }

        # Split 80 / 20
        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_set, test_set = torch.utils.data.random_split(
            full_dataset, [train_size, test_size]
        )

        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])
        test_set.dataset.transform = test_transform

        train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

        # -------- alameer --------
        self.model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )
        self.model.classifier[1] = nn.Linear(
            self.model.classifier[1].in_features,
            num_classes
        )
        self.model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

        # -------- Training --------
        epochs = 10
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)
            self.status_label.config(
                text=f"Training... Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}"
            )

        # -------- Migo --------
        self.model.eval()
        correct = total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        accuracy = 100 * correct / total
        self.acc_label.config(text=f"Model Ready (Acc: {accuracy:.1f}%)")

        torch.save(self.model.state_dict(), "potato_deep_model.pth")

        self.status_label.config(text="Training Complete ✅")
        self.train_btn.config(state="normal")
        messagebox.showinfo(
            "Success",
            f"Deep Learning Model Trained!\nAccuracy: {accuracy:.1f}%"
        )

    def load_saved_model(self):
        checkpoint = torch.load(
            "potato_deep_model.pth",
            map_location=self.device
        )

        num_classes = checkpoint["classifier.1.weight"].shape[0]

        self.model = models.efficientnet_b0(weights=None)
        self.model.classifier[1] = nn.Linear(
            self.model.classifier[1].in_features,
            num_classes
        )

        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()

        self.label_map = {i: f"Class {i}" for i in range(num_classes)}

        self.status_label.config(text="Pre-trained Model Loaded")
        self.acc_label.config(text="Model Ready")

    # ================= bondok =================
    def detect_disease(self):
        if not self.model:
            messagebox.showerror("Error", "Train or load model first!")
            return

        path = filedialog.askopenfilename(title="Select Leaf Image")
        if not path:
            return

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

        img = Image.open(path).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)
            probs = torch.softmax(output, dim=1)[0]
            conf = probs.max().item()
            idx = probs.argmax().item()

        if conf < self.CONFIDENCE_THRESHOLD:
            text = "Unknown / Non-Potato Image"
            color = "#FFA500"
        else:
            name = self.label_map.get(idx, "Unknown")
            text = f"{name} ({conf*100:.1f}%)"
            color = "#00FF00" if "Healthy" in name else "#FF4444"

        self.result_label.config(text=text, fg=color)

        img = ImageEnhance.Contrast(img).enhance(1.3)
        img.thumbnail((700, 600))
        img_tk = ImageTk.PhotoImage(img)
        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk


if __name__ == "__main__":
    root = tk.Tk()
    app = PotatoLeafDeepLearner(root)
    root.mainloop()
