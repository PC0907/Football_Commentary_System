---

# **Jersey Image Annotator Tool**  
A powerful **image annotation tool** designed for labeling **football jersey numbers**. It supports **manual annotation** and **automatic augmentation**, making it ideal for training OCR models on jersey numbers.

---

## **📌 Features**
✔ **Manual Annotation** – Quickly label jersey numbers using keyboard input.  
✔ **Augmentation Mode** – Generates **augmented versions** of images for better OCR training.  
✔ **Session Management** – Saves and **resumes from the last annotated image**.  
✔ **CSV Export** – Saves labels in a structured CSV format for model training.  
✔ **Keyboard Shortcuts** – Navigate, label, and save annotations efficiently.  
✔ **Intuitive UI** – A **modern, dark-themed interface** for easy annotation.

---

## **📂 Project Structure**
```
JerseyImageAnnotator/
│── annotator.py        # Main GUI application
│── image_loader.py     # Handles loading & navigation of images
│── csv_handler.py      # Saves annotations in a CSV file
│── augmentor.py        # Applies augmentation techniques
│── session_data.json   # Stores session progress (auto-generated)
│── annotations.csv     # Stores labeled data (auto-generated)
│── assets/             # Icons, UI assets (optional)
│── README.md           # Documentation
```

---

## **⚙️ Installation**
### **1️⃣ Install Dependencies**
Ensure you have Python **3.8+** installed. Then install dependencies:

```bash
pip install opencv-python numpy PyQt5
```

### **2️⃣ Clone the Repository**
```bash
git clone https://github.com/yourusername/JerseyImageAnnotator.git
cd JerseyImageAnnotator
```

### **3️⃣ Run the Application**
```bash
python annotator.py
```

---

## **🎨 User Interface**
### **🖥️ Main UI Components**
- **📷 Image Display** – Shows the current image being labeled.
- **🔢 Number Display** – Shows the currently entered number.
- **🎛️ Control Buttons** – Navigate, save, and toggle augmentation.
- **📂 Folder Selection** – Load images & set output directory.
- **💾 Session Management** – Automatically resumes previous sessions.

---

## **🛠️ How to Use**
### **1️⃣ Load Images**
- Click **📂 Load Folder** to select a directory with images.
- Click **📁 Select Output Folder** to choose where annotated images and CSV files will be saved.

### **2️⃣ Label Images**
- **Type the jersey number** using the keyboard.
- Press **Enter** to save the label.

### **3️⃣ Navigate Between Images**
- **← Left Arrow**: Go to the previous image.
- **→ Right Arrow**: Go to the next image.

### **4️⃣ Augment Images (Optional)**
- Enable **"Augmentation Mode"** to generate **10+ augmented variations** per image.
- The original image is saved as `_aug0`, augmented images as `_aug1`, `_aug2`, etc.

### **5️⃣ Resume Previous Session**
- If a previous session exists, a **popup notification** will inform you when resuming.

---

## **🎯 Keyboard Shortcuts**
| Key | Action |
|-----|--------|
| **Left Arrow (←)** | Go to the previous image |
| **Right Arrow (→)** | Go to the next image |
| **0-9 Keys** | Enter jersey number |
| **Backspace** | Delete last digit |
| **Enter** | Save annotation & move to next image |

---

## **📜 Output Files**
### **1️⃣ CSV File (`annotations.csv`)**
Stores annotations in the format:

| image_name | label | session_id | timestamp |
|------------|-------|------------|------------|
| `IMG_0001.jpg` | `10` | `20240225_1405` | `2024-02-25 14:05:32` |
| `IMG_0001_aug1.jpg` | `10` | `20240225_1405` | `2024-02-25 14:05:35` |

### **2️⃣ Augmented Images**
Saved in the **output folder** as:
```
IMG_0001_aug0.jpg  # Original image
IMG_0001_aug1.jpg  # Augmented version 1
IMG_0001_aug2.jpg  # Augmented version 2
...
```

### **3️⃣ Session Data (`session_data.json`)**
Tracks progress so you can **resume labeling from where you left off**.

---

## **🐞 Troubleshooting**
### **1️⃣ Images Not Saving?**
- Ensure the **output folder** is selected.
- Check for errors in the **terminal log** (run with `python annotator.py`).
- Manually **set correct folder permissions**:
  ```bash
  chmod -R 777 /path/to/output/folder/
  ```

### **2️⃣ Augmented Images Not Appearing?**
- Make sure **"Augmentation Mode"** is **enabled** before saving.
- Check if `cv2.imwrite()` is failing by looking for **error messages** in the terminal.

### **3️⃣ Resuming a Previous Session Fails?**
- Ensure `session_data.json` and `annotations.csv` are in the **output folder**.
- Restart the tool and check for a **popup notification** confirming session resume.

---

## **🚀 Future Improvements**
- **[ ] Auto-detect jersey numbers using OCR**  
- **[ ] Support for multiple players per image**  
- **[ ] Real-time annotation mode (faster navigation)**  

---

## **👨‍💻 Author & Contributions**
- **Created by:** *Fawwaz Bin Tasneem, Syed Ali Mehdi Rizvi as a part of our Final Year B.Tech Project*  
- **Contributions:** Open a pull request on GitHub!  

---

## **📜 License**
This project is **open-source** under the **MIT License**.

---
