import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk

class SmoothRegionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Smooth Region")

        self.image = None
        self.image_temp = None
        self.image_smoothed = None
        self.selected_regions = []
        self.filter_type = tk.StringVar()
        self.filter_parameter = tk.StringVar()
        self.mode = tk.StringVar(value="Manual")  # Default selection: Manual mode
        self.smooth_executed = False  # Adding a variable to track whether 'Smooth Region' has been executed once under "Automatic" mode

        # Create buttons frame
        self.buttons_frame = tk.Frame(master)
        self.buttons_frame.pack(side=tk.TOP, padx=5, pady=5, fill=tk.X)

        # Create buttons
        self.btn_open = tk.Button(self.buttons_frame, text="Open Image", command=self.open_image)
        self.btn_open.pack(side=tk.LEFT, padx=5, pady=5)
        self.btn_smooth = tk.Button(self.buttons_frame, text="Smooth Region", command=self.smooth_region)
        self.btn_smooth.pack(side=tk.LEFT, padx=5, pady=5)
        self.btn_undo = tk.Button(self.buttons_frame, text="Undo", command=self.undo)
        self.btn_undo.pack(side=tk.LEFT, padx=5, pady=5)

        # Create filter frame
        self.filter_frame = tk.Frame(master)
        self.filter_frame.pack(side=tk.TOP, padx=5, pady=5, fill=tk.X)

        # Create dropdown menu for filter selection
        self.filter_label = tk.Label(self.filter_frame, text="Select Filter:")
        self.filter_label.pack(side=tk.LEFT, padx=5, pady=5, anchor=tk.W)
        self.filter_menu = ttk.Combobox(self.filter_frame, textvariable=self.filter_type, values=["Ideal Low Pass", "Butterworth Low Pass", "Gaussian Low Pass"])
        self.filter_menu.pack(side=tk.LEFT, padx=5, pady=5)
        self.filter_menu.current(0)  # Default selection: the first filter
        self.filter_menu.bind("<<ComboboxSelected>>", self.update_parameter_entry)

        # Create parameter frame
        self.parameter_frame = tk.Frame(master)
        self.parameter_frame.pack(side=tk.TOP, padx=5, pady=5, fill=tk.X)

        # Create parameter entry
        self.parameter_label = tk.Label(self.parameter_frame, text="Filter Parameter:")
        self.parameter_label.pack(side=tk.LEFT, padx=5, pady=5, anchor=tk.W)
        self.parameter_entry = tk.Entry(self.parameter_frame, textvariable=self.filter_parameter)
        self.parameter_entry.pack(side=tk.LEFT, padx=5, pady=5)
        self.filter_parameter.set("5")  # Default parameter

        # Create order frame
        self.order_frame = tk.Frame(master)
        self.order_frame.pack(side=tk.TOP, padx=5, pady=5, fill=tk.X)

        # Create order entry
        self.order_label = tk.Label(self.order_frame, text="Order:")
        self.order_label.pack(side=tk.LEFT, padx=5, pady=5, anchor=tk.W)
        self.order_entry = tk.Entry(self.order_frame)
        self.order_entry.pack(side=tk.LEFT, padx=5, pady=5)
        self.order_entry.insert(tk.END, "5")  # Default order: 5
        self.order_label.config(state=tk.DISABLED)
        self.order_entry.config(state=tk.DISABLED)

        # Create mode frame
        self.mode_frame = tk.Frame(master)
        self.mode_frame.pack(side=tk.TOP, padx=5, pady=5, fill=tk.X)

        # Create mode selection dropdown
        self.mode_label = tk.Label(self.mode_frame, text="Select Mode:")
        self.mode_label.pack(side=tk.LEFT, padx=5, pady=5, anchor=tk.W)
        self.mode_menu = ttk.Combobox(self.mode_frame, textvariable=self.mode, values=["Manual", "Automatic"])
        self.mode_menu.pack(side=tk.LEFT, padx=5, pady=5)

        # Canvas for image display
        self.canvas = tk.Canvas(master, width=640, height=480)
        self.canvas.pack()
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)



    def open_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # Read the image in grayscale mode.
            self.image_temp = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            self.display_image(self.image)
        self.smooth_executed = False

    def display_image(self, image):
        if image is not None:
            image_pil = Image.fromarray(image)
            image_tk = ImageTk.PhotoImage(image_pil)
            self.canvas.image = image_tk
            self.canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)

    def on_press(self, event):
        self.rect_start_x = event.x
        self.rect_start_y = event.y

    def on_drag(self, event):
        self.rect_end_x = event.x
        self.rect_end_y = event.y
        self.canvas.delete("rect")
        self.canvas.create_rectangle(self.rect_start_x, self.rect_start_y,
                                     self.rect_end_x, self.rect_end_y,
                                     outline="red", tags="rect")

    def on_release(self, event):
        self.rect_end_x = event.x
        self.rect_end_y = event.y
        self.selected_regions.append((self.rect_start_x, self.rect_start_y, self.rect_end_x, self.rect_end_y))

    def smooth_region(self):
        if self.image is None:
            return

        # Add a condition to check if "Smooth Region" has been executed in 'Automatic' mode once, then do not perform any operation.
        if self.mode.get() == "Automatic" and self.smooth_executed:
            return

        self.image_smoothed = self.image.copy()

        filter_type = self.filter_type.get()
        filter_parameter = float(self.filter_parameter.get())

        if self.mode.get() == "Manual":
            for region in self.selected_regions:
                x1, y1, x2, y2 = region
                region_image = self.image_smoothed[y1:y2, x1:x2]

                if filter_type == "Ideal Low Pass":
                    smoothed_region = ideal_low_pass_filter(region_image, filter_parameter)
                elif filter_type == "Butterworth Low Pass":
                    order = int(self.order_entry.get())
                    smoothed_region = Butterworth_low_pass_filter(region_image, filter_parameter, order)
                elif filter_type == "Gaussian Low Pass":
                    smoothed_region = Gaussian_low_pass_filter(region_image, filter_parameter)
                self.image_smoothed[y1:y2, x1:x2] = smoothed_region
        elif self.mode.get() == "Automatic":
            if filter_type == "Butterworth Low Pass":
                order = int(self.order_entry.get())
                self.image_smoothed = detect_wrinkles(self.image,filter_parameter,order,filter_type)
            else:
                self.image_smoothed = detect_wrinkles(self.image,filter_parameter,0,filter_type)
            # Mark that "Smooth Region" has been executed once in 'Automatic' mode.
            self.smooth_executed = True
        self.display_image(self.image_smoothed)

    def update_parameter_entry(self, event):
        selected_filter = self.filter_type.get()
        selected_mode = self.mode.get()
        if selected_filter == "Butterworth Low Pass":
            self.order_label.config(state=tk.NORMAL)
            self.order_entry.config(state=tk.NORMAL)
        else:
            self.order_label.config(state=tk.DISABLED)
            self.order_entry.config(state=tk.DISABLED)

    def undo(self):
        self.smooth_executed = False
        if self.mode.get() == 'Automatic':
            self.smooth_executed = False
            self.image = self.image_temp.copy()
            self.display_image(self.image)
        else:
            if len(self.selected_regions) > 0:
                # Remote the last element in chart
                self.selected_regions.pop()

                # Check whether have the image
                if self.image is not None:
                    self.image_smoothed = self.image.copy()

                    filter_type = self.filter_type.get()
                    filter_parameter = float(self.filter_parameter.get())

                    if self.mode.get() == "Manual":
                        # Reapply the filter to each selected region.
                        for region in self.selected_regions:
                            x1, y1, x2, y2 = region
                            region_image = self.image_smoothed[y1:y2, x1:x2]

                            if filter_type == "Ideal Low Pass":
                                smoothed_region = ideal_low_pass_filter(region_image, filter_parameter)
                            elif filter_type == "Butterworth Low Pass":
                                order = int(self.order_entry.get())
                                smoothed_region = Butterworth_low_pass_filter(region_image, filter_parameter, order)
                            elif filter_type == "Gaussian Low Pass":
                                smoothed_region = Gaussian_low_pass_filter(region_image, filter_parameter)

                            self.image_smoothed[y1:y2, x1:x2] = smoothed_region
            if self.image_smoothed is not None:
                self.display_image(self.image_smoothed)
            else:
                self.display_image(self.image)  # Display the original image when there is no processed result.
         

def detect_wrinkles(image, filter_parameter, order, filter_type):
    if len(image.shape) == 3 and image.shape[2] > 1:
        # If the image has multiple channels, convert it to a grayscale image.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Load face and eye detection models.
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    # Detect the face
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # If detect the face
    if len(faces) > 0:
        for (fx, fy, fw, fh) in faces:
            # Process the wrinkle detection in the face region
            face_roi = gray[fy:fy+int(3*fh/5), fx+int(2*fw/11):fx+int(9*fw/11)]

            # Load eye detection model
            eyes = eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=5)

            # 绘制矩形框选区域并获取眼睛和嘴巴检测区域
            wrinkle_region = np.ones_like(face_roi) * 255
            for (ex, ey, ew, eh) in eyes:
                # 绘制矩形框选区域（眼睛上方）
                cv2.rectangle(wrinkle_region, (ex, ey + int(6 * eh / 9)), (ex + ew, ey - int(3 * eh / 9)), (0, 0, 0), -1)

            # Perform edge detection using the Sobel operator.
            sobel_x = cv2.Sobel(face_roi, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(face_roi, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

            # Sobel edge image
            wrinkle_image = np.uint8(gradient_magnitude > 100) * 255
            wrinkle_image = cv2.bitwise_and(wrinkle_image, wrinkle_region)

            # Find the contours of detected wrinkle areas
            contours, _ = cv2.findContours(wrinkle_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Draw the detected wrinkle areas on the original image
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                # cv2.rectangle(face_roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Apply Gaussian filtering to the wrinkle areas.
                face_roi[y:y+h, x:x+w] = cv2.GaussianBlur(face_roi[y:y+h, x:x+w], (21, 21), 0)
            # 将处理后的人脸区域放回原图中
            gray[fy:fy+int(3*fh/5), fx+int(2*fw/11):fx+int(9*fw/11)] = face_roi
    else:
        # If no face is detected, perform wrinkle detection on the entire image area.
        # Draw a rectangular box to select the area and obtain the area for eye detection.
        wrinkle_region = np.ones_like(gray) * 255
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (ex, ey, ew, eh) in eyes:
            # Draw a rectangular box to select the region (above the eyes).
            cv2.rectangle(wrinkle_region, (ex, ey + int(6 * eh / 9)), (ex + ew, ey - int(3 * eh / 9)), (0, 0, 0), -1)

        # Perform edge detection using the Sobel operator.
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

        # Sobel edeg image
        wrinkle_image = np.uint8(gradient_magnitude > 100) * 255

        # Perform wrinkle detection outside the eyes
        wrinkle_image = cv2.bitwise_and(wrinkle_image, wrinkle_region)

        # Find the contours of detected wrinkle areas
        contours, _ = cv2.findContours(wrinkle_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Process the filter to the selected region
            if filter_type == "Ideal Low Pass":
                gray[y:y+h, x:x+w] = ideal_low_pass_filter(gray[y:y+h, x:x+w], filter_parameter)
            elif filter_type == "Butterworth Low Pass":
                gray[y:y+h, x:x+w] = Butterworth_low_pass_filter(gray[y:y+h, x:x+w], filter_parameter, order)
            elif filter_type == "Gaussian Low Pass":
                gray[y:y+h, x:x+w] = Gaussian_low_pass_filter(gray[y:y+h, x:x+w], filter_parameter)
    return gray

def ideal_low_pass_filter(img, d0):
    dft = np.fft.fft2(img)
    dft_shift = np.fft.fftshift(dft)
    rows, cols = img.shape
    crow, ccol = int(rows/2), int(cols/2)
    mask = np.zeros((rows, cols), np.uint8)
    for x in range(0,rows):
        for y in range(0,cols):
            d = np.sqrt((crow-x)**2+(ccol-y)**2)
            if d <= d0:
                mask[x,y] = 1
    dft_shift *= mask
    f_ishift = np.fft.ifftshift(dft_shift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return img_back.astype(np.uint8)

def Butterworth_low_pass_filter(img, d0, n):
    dft = np.fft.fft2(img)
    dft_shift = np.fft.fftshift(dft)
    rows, cols = img.shape
    crow, ccol = int(rows/2), int(cols/2)
    mask = np.zeros((rows, cols))
    for x in range(rows):
        for y in range(cols):
            d = np.sqrt((x-crow)**2 + (y-ccol)**2)
            mask[x, y] = 1 / (1 + (d/d0)**(2*n))
    dft_shift *= mask
    f_ishift = np.fft.ifftshift(dft_shift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return img_back.astype(np.uint8)

def Gaussian_low_pass_filter(img, d0):
    dft = np.fft.fft2(img)
    dft_shift = np.fft.fftshift(dft)
    rows, cols = img.shape
    crow, ccol = int(rows/2), int(cols/2)
    mask = np.zeros((rows, cols))
    for x in range(rows):
        for y in range(cols):
            d = (x - crow)**2 + (y - ccol)**2
            mask[x, y] = np.exp(-d / (2 * d0**2))
    dft_shift *= mask
    f_ishift = np.fft.ifftshift(dft_shift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return img_back.astype(np.uint8)

def main():
    root = tk.Tk()
    app = SmoothRegionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
