import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from typing import Dict, Tuple, List, Optional
from scipy.ndimage import gaussian_filter1d

class thwImageProcessorGPU:
    def __init__(self, config: dict):
        self.config = config
        self.images = {}
        self.z_values = []
        self.result_image = None
        self.cuda_module = self._compile_cuda_kernel()
        self.calculate_thw_kernel = self.cuda_module.get_function("calculatethw")

    def _compile_cuda_kernel(self):
        return SourceModule("""
        __global__ void calculatethw(
            unsigned char* images, 
            int* z_values, 
            int center_z, 
            int max_thw,
            int num_images, 
            int height, 
            int width,
            unsigned char* result
        ) {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;
            if (x >= width || y >= height) return;

            int pixel_idx = y * width + x;
            int center_index = -1;
            for (int i = 0; i < num_images; i++) {
                if (z_values[i] == center_z) {
                    center_index = i;
                    break;
                }
            }
            if (center_index == -1) {
                result[pixel_idx] = max_thw;
                return;
            }

            float curve[102];
            for (int i = 0; i < num_images; i++) {
                curve[i] = images[i * height * width + pixel_idx];
            }

            
            float gauss_kernel[102];
            float sigma = 1;
            int ksize = 3;
            float sum = 0.0;
            int half = ksize / 2;
            for (int i = 0; i < ksize; i++) {
                float x = i - half;
                float g = expf(-0.5f * (x * x) / (sigma * sigma));
                gauss_kernel[i] = g;
                sum += g;
            }
            for (int i = 0; i < ksize; i++) {
                gauss_kernel[i] /= sum;
            }

            float smooth[102];
            for (int i = 0; i < num_images; i++) {
                float val = 0.0f;
                for (int j = 0; j < ksize; j++) {
                    int idx = i + j - half;
                    if (idx < 0) idx = 0;
                    if (idx >= num_images) idx = num_images - 1;
                    val += curve[idx] * gauss_kernel[j];
                }
                smooth[i] = val;
            }
                            

            float diffs[102];
            for (int i = 0; i < num_images - 1; i++) {
                diffs[i] = smooth[i + 1] - smooth[i];
            }

            
            int local_min_indices[102];
            int min_count = 0;
            for (int i = 0; i < num_images - 2; i++) {
                float sign1 = 0.0f, sign2 = 0.0f;
                if (diffs[i] > 0) sign1 = 1.0f;
                else if (diffs[i] < 0) sign1 = -1.0f;
                // diffs[i] == 0 时 sign1 = 0

                if (diffs[i+1] > 0) sign2 = 1.0f;
                else if (diffs[i+1] < 0) sign2 = -1.0f;

                if (sign1 != sign2) {
                    local_min_indices[min_count++] = i + 1;
                }
            }             

            int left_min = -1;
            int right_min = -1;
            for (int i = 0; i < min_count; i++) {
                if (local_min_indices[i] < center_index) {
                    left_min = local_min_indices[i];
                }
            }
            for (int i = 0; i < min_count; i++) {
                if (local_min_indices[i] > center_index) {
                    right_min = local_min_indices[i];
                    break;
                }
            }

            int half_peak_width = 0;
            if (left_min != -1 && right_min != -1) {
                half_peak_width = right_min - left_min;
            }
            if (half_peak_width <= 5) {     
                half_peak_width = 0;
            }
            if (half_peak_width > max_thw) {
                half_peak_width = max_thw;
            }
            result[pixel_idx] = half_peak_width / 2;
        }
        """)

    def load_images(self) -> None:
        folder = self.config['folder_path']
        x, y, width, height = self.config['roi']

        for filename in os.listdir(folder):
            if filename.endswith(".bmp"):
                try:
                    img_path = os.path.join(folder, filename)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    img_num = int(os.path.splitext(filename)[0])
                    cropped_img = img[y:y+height, x:x+width]
                    self.images[img_num] = cropped_img
                except Exception as e:
                    print(f"Error loading {filename}: {e}")

        self.z_values = np.array(sorted(self.images.keys()), dtype=np.int32)
        print(f"Loaded {len(self.images)} images")

    def process_images(self) -> None:
        if not self.images:
            raise ValueError("No images loaded. Call load_images() first.")

        height, width = next(iter(self.images.values())).shape
        num_images = len(self.images)
        center_z = self.config['anchor_z']
        max_thw = self.config['max_thw']

        images_array = np.zeros((num_images, height, width), dtype=np.uint8)
        for i, z in enumerate(self.z_values):
            images_array[i] = self.images[z]

        flat_images = images_array.flatten()
        result_host = np.zeros((height, width), dtype=np.uint8)

        images_gpu = cuda.mem_alloc(flat_images.nbytes)
        z_values_gpu = cuda.mem_alloc(self.z_values.nbytes)
        result_gpu = cuda.mem_alloc(result_host.nbytes)

        cuda.memcpy_htod(images_gpu, flat_images)
        cuda.memcpy_htod(z_values_gpu, self.z_values)

        block_size = (16, 16, 1)
        grid_size = ((width + block_size[0] - 1) // block_size[0], 
                     (height + block_size[1] - 1) // block_size[1])

        self.calculate_thw_kernel(
            images_gpu, z_values_gpu, np.int32(center_z), np.int32(max_thw),
            np.int32(num_images), np.int32(height), np.int32(width),
            result_gpu,
            block=block_size, grid=grid_size
        )

        cuda.memcpy_dtoh(result_host, result_gpu)
        self.result_image = cv2.medianBlur(result_host, 3)

    def save_result(self, output_path: Optional[str] = None) -> None:
        if self.result_image is None:
            raise ValueError("No processed image available. Call process_images() first.")

        output_path = output_path or self.config.get('output_path', 'thw_result.bmp')
        cv2.imwrite(output_path, self.result_image)
        print(f"Result saved to {output_path}")


def main():
    config = {
        'folder_path': 'D:/dataset/80min',
        'roi': (33, 50, 620, 970),
        'anchor_z': -110,
        'max_thw': 90,
        'output_path': 'D:/dataset/80min/result.bmp'
    }

    processor = thwImageProcessorGPU(config)
    processor.load_images()

    start_time = time.time()
    processor.process_images()
    print(f"Time cost at processing images: {time.time() - start_time:.2f}s")
    
    processor.save_result()

if __name__ == "__main__":
    main()