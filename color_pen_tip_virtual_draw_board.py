import cv2                     
import numpy as np             

# Tạo canvas trắng để vẽ, nó là bước khởi tạo ban đầu
canvas = None  # Gía trị canvas rỗng vì chưa có gì để vẽ

# Định nghĩa các màu sắc cần nhận diện.
# Màu sắc được định nghĩa theo không gian màu HSV.
# HSV (Hue, Saturation, Value) là không gian màu thường được sử dụng trong xử lý ảnh.
# Hue (sắc độ), Saturation (độ bão hòa), Value (độ sáng).
# vì nó dễ dàng phân biệt các màu sắc hơn so với không gian màu RGB.
# Các ngưỡng màu sắc được định nghĩa dưới dạng numpy array.
# Màu vẽ tương ứng ngưỡng và với từng đầu bút.
color_ranges = {
    "yellow": {
        "lower": np.array([15, 80, 80]),    # giá trị thấp nhất (ngưỡng dưới) HSV cho màu vàng
        "upper": np.array([35, 255, 255]),  # giá trị cao nhất (ngưỡng trên) HSV cho màu vàng
        "draw_color": (0, 255, 255)         # Màu vẽ vàng (BGR)
    },
    "pink": {
        "lower": np.array([140, 50, 50]),   # giá trị thấp nhất (ngưỡng dưới) HSV cho màu hồng
        "upper": np.array([180, 255, 255]),  # giá trị cao nhất (ngưỡng trên) HSV cho màu hồng
        "draw_color": (255, 0, 255)          # Màu vẽ hồng (BGR)
    },
    "green": {
        "lower": np.array([35, 100, 100]),   # giá trị thấp nhất (ngưỡng dưới) HSV cho màu xanh lá
        "upper": np.array([85, 255, 255]),  # giá trị cao nhất (ngưỡng trên) HSV cho màu xanh lá
        "draw_color": (0, 255, 0)           # Màu vẽ xanh lá (BGR)
    }
}

# Tọa độ trước đó cho từng màu
prev_points = {
    "yellow": None,  # Lưu vị trí trước đó của đầu bút vàng
    "pink": None,  # Lưu vị trí trước đó của đầu bút hồng
    "green": None # Lưu vị trí trước đó của đầu bút xanh lá
}

# Khởi tạo webcam (Image Acquisition - Thu nhận ảnh)
cap = cv2.VideoCapture(0)      # Mở webcam (0 = webcam mặc định, 1 = webcam thứ 2 nếu có)
cv2.namedWindow("Virtual Drawing Board")  # Tạo cửa sổ hiển thị

while True:                    # Vòng lặp chính
    ret, frame = cap.read()  # Đọc một khung hình từ webcam
    if not ret:             # Nếu không đọc được thì thoát
        break
    # Lật ảnh theo chiều ngang giống gương
    frame = cv2.flip(frame, 1) 
    print(frame.shape)

    if canvas is None:         # Nếu chưa có canvas thì tạo canvas trắng cùng kích thước frame
        canvas = np.zeros_like(frame) # Canvas sẽ có kích thước và số kênh màu giống hệt với khung ảnh lấy từ camera.

    # Preprocessing (Tiền xử lý)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Chuyển ảnh từ BGR sang HSV

    for color_name, params in color_ranges.items():   # Lặp qua từng màu cần nhận diện
        lower_color = params["lower"]                 # Ngưỡng dưới HSV
        upper_color = params["upper"]                 # Ngưỡng trên HSV
        draw_color = params["draw_color"]             # Màu vẽ

        mask = cv2.inRange(hsv, lower_color, upper_color)  # Tạo mask cho vùng màu cần nhận diện
        mask = cv2.erode(mask, None, iterations=2)         # Xóa nhiễu nhỏ bằng phép co
        mask = cv2.dilate(mask, None, iterations=2)        # Làm đầy vùng bằng phép giãn
        #_ là giá trị trả về không cần thiết
        # contours: danh sách các đường viền tìm thấy trong ảnh
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
        # cv2.RETR_EXTERNAL: chỉ lấy các đường viền bên ngoài
        # cv2.CHAIN_APPROX_SIMPLE: giảm số lượng điểm trong đường viền

###Rút trích đặc trưng của contour
#contourArea: đặc trưng về kích thước.
#minEnclosingCircle: đặc trưng về hình dạng và vị trí.
#center: đặc trưng về tọa độ.
#radius: đặc trưng về kích cỡ.

        # Detect/Recognition (Phát hiện/Nhận diện)
        if contours:   #contours ở đây là đường viền
                                                               # Nếu tìm thấy contour
            largest_contour = max(contours, key=cv2.contourArea)          # Lấy contour lớn nhất
            if cv2.contourArea(largest_contour) > 3000:                   # Nếu diện tích đủ lớn (loại bỏ nhiễu)
                (x, y), radius = cv2.minEnclosingCircle(largest_contour)  # Tìm tâm và bán kính hình tròn ngoại tiếp nhỏ nhất
                center = (int(x), int(y))                                 # Lấy tọa độ tâm

                # Vẽ vòng tròn nhận diện đầu bút
                cv2.circle(frame, center, int(radius), draw_color, 2) # Vẽ vòng tròn quanh đầu bút, 2 ở đây là độ dày của đường tròn

                # Vẽ nét (Decision/Output)
                if prev_points[color_name] is not None:          # Nếu đã có vị trí trước đó
                    cv2.line(canvas, prev_points[color_name], center, draw_color, 5)  # Vẽ đường nối, 5 là độ dày của đường vẽ
                prev_points[color_name] = center                # Cập nhật vị trí hiện tại
            else:
                prev_points[color_name] = None                 # Nếu contour nhỏ, bỏ qua
        else:
            prev_points[color_name] = None                    # Nếu không tìm thấy contour, bỏ qua

    # Hiển thị tổng hợp hình
    ogriwithcans = cv2.add(frame, canvas)              # Kết hợp ảnh gốc với canvas vẽ
    cv2.imshow("Virtual Drawing Board", ogriwithcans)  # Hiển thị lên cửa sổ

    key = cv2.waitKey(1) & 0xFF    # Đợi phím nhấn (OpenCV), 0xFF là số 255 trong hex
    if key == ord('a'):               # Nếu nhấn 'a' thì xóa canvas
        canvas = np.zeros_like(frame)
    elif key == 27:                       # Nếu nhấn ESC thì thoát vòng lặp
        break
cap.release()              # Giải phóng webcam (OpenCV)
cv2.destroyAllWindows()    # Đóng tất cả cửa sổ hiển thị (OpenCV)