import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk

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

class SmoothRegionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Smooth Region")
        
        self.image = None
        self.image_smoothed = None
        self.rect_start_x = None
        self.rect_start_y = None
        self.rect_end_x = None
        self.rect_end_y = None
        self.selected_regions = []
        self.filter_type = tk.StringVar()
        self.filter_parameter = tk.StringVar()
        
        # Create buttons
        self.btn_open = tk.Button(master, text="Open Image", command=self.open_image)
        self.btn_open.pack(side=tk.LEFT, padx=5, pady=5)
        self.btn_smooth = tk.Button(master, text="Smooth Region", command=self.smooth_region)
        self.btn_smooth.pack(side=tk.LEFT, padx=5, pady=5)
        self.btn_undo = tk.Button(master, text="Undo", command=self.undo)
        self.btn_undo.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Create dropdown menu
        self.filter_label = tk.Label(master, text="Select Filter:")
        self.filter_label.pack(side=tk.LEFT, padx=5, pady=5)
        self.filter_menu = ttk.Combobox(master, textvariable=self.filter_type, values=["Ideal Low Pass", "Butterworth Low Pass", "Gaussian Low Pass"])
        self.filter_menu.pack(side=tk.LEFT, padx=5, pady=5)
        self.filter_menu.current(0)  # 默认选择第一个滤波器
        self.filter_menu.bind("<<ComboboxSelected>>", self.update_parameter_entry)
        
        # Create parameter entry
        self.parameter_label = tk.Label(master, text="Filter Parameter:")
        self.parameter_label.pack(side=tk.LEFT, padx=5, pady=5)
        self.parameter_entry = tk.Entry(master, textvariable=self.filter_parameter)
        self.parameter_entry.pack(side=tk.LEFT, padx=5, pady=5)
        self.filter_parameter.set("5")  # 默认参数
        
        self.order_label = tk.Label(master, text="Order:")
        self.order_label.pack(side=tk.LEFT, padx=5, pady=5)
        self.order_entry = tk.Entry(master)
        self.order_entry.pack(side=tk.LEFT, padx=5, pady=5)
        self.order_entry.insert(tk.END, "5")  # 默认阶数为5
        self.order_label.config(state=tk.DISABLED)
        self.order_entry.config(state=tk.DISABLED)

        # Canvas for image display
        self.canvas = tk.Canvas(master, width=640, height=480)
        self.canvas.pack()
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
    
    def open_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # 以灰度模式读取图像
            self.display_image(self.image)

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
        
        if len(self.selected_regions) == 0:
            return
        
        self.image_smoothed = self.image.copy()
        
        filter_type = self.filter_type.get()
        filter_parameter = float(self.filter_parameter.get())
        order = int(self.order_entry.get()) if filter_type == "Butterworth Low Pass" else 0
        
        # 设置膨胀和腐蚀操作的内核大小
        kernel_size = 5
        kernel_dilate = np.ones((kernel_size, kernel_size), np.uint8)
        kernel_erode = np.ones((kernel_size, kernel_size), np.uint8)
        
        for region in self.selected_regions:
            x1, y1, x2, y2 = region
            region_image = self.image_smoothed[y1:y2, x1:x2]
            
            # 对选定区域进行膨胀和腐蚀操作
            region_image = cv2.dilate(region_image, kernel_dilate, iterations=1)
            region_image = cv2.erode(region_image, kernel_erode, iterations=1)
            
            if filter_type == "Ideal Low Pass":
                smoothed_region = ideal_low_pass_filter(region_image, filter_parameter)
            elif filter_type == "Butterworth Low Pass":
                smoothed_region = Butterworth_low_pass_filter(region_image, filter_parameter, order)
            elif filter_type == "Gaussian Low Pass":
                smoothed_region = Gaussian_low_pass_filter(region_image, filter_parameter)
            
            self.image_smoothed[y1:y2, x1:x2] = smoothed_region
        
        self.display_image(self.image_smoothed)


    
    def update_parameter_entry(self, event):
        selected_filter = self.filter_type.get()
        if selected_filter == "Butterworth Low Pass":
            self.order_label.config(state=tk.NORMAL)
            self.order_entry.config(state=tk.NORMAL)
        else:
            self.parameter_label.config(text="Filter Parameter:")
            self.order_label.config(state=tk.DISABLED)
            self.order_entry.config(state=tk.DISABLED)
    
    def undo(self):
        if len(self.selected_regions) > 0:
            # 清空已选择的区域
            self.selected_regions.pop()
            
            # 检查是否有原始图像
            if self.image is not None:
                self.image_smoothed = self.image.copy()
                filter_type = self.filter_type.get()
                filter_parameter = float(self.filter_parameter.get())
                
                # 对每个选定的区域重新应用滤波器
                for region in self.selected_regions:
                    x1, y1, x2, y2 = region
                    region_image = self.image_smoothed[y1:y2, x1:x2]
                    
                    if filter_type == "Ideal Low Pass":
                        smoothed_region = ideal_low_pass_filter(region_image, filter_parameter)
                    elif filter_type == "Butterworth Low Pass":
                        smoothed_region = Butterworth_low_pass_filter(region_image, filter_parameter,5)
                    elif filter_type == "Gaussian Low Pass":
                        smoothed_region = Gaussian_low_pass_filter(region_image, filter_parameter)
                    self.image_smoothed[y1:y2, x1:x2] = smoothed_region
            
            # 更新显示的图像
            if self.image_smoothed is not None:
                self.display_image(self.image_smoothed)
            else:
                self.display_image(self.image)  # 没有处理结果时显示原始图像

def main():
    root = tk.Tk()
    app = SmoothRegionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
