"""
    Created by Adam Kohl
"""
# Import helper methods
from ultralytics import YOLO
from img_preprocessor import *
# Import modules
import json
import socket
import hashlib
import sys

"""
    ENABLE FOR TESTING
"""
DEBUG_FLAG = False

"""
    NETWORKING
"""
TCP_IP = "127.0.0.1"
TCP_PORT = 5005  # Fixed port for single AV
BUFFER_SIZE = 65536
MODEL_IDX = 0  # Fixed model index for single AV


model = None


def load_model():
    global model
    try:
        # Load a pretrained YOLO11n model
        model = YOLO("yolo11n.pt")
        print()
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


def process_image(img_bytes):
    try:
        print()
    except Exception as e:
        print(f"Error processing image: {e}")
        raise


def detect_objects(image, model):
    """Run YOLOv8 object detection and return bounding box info."""
    # Run inference
    results = model(image, verbose=False)[0]  # Get first result (single image)

    boxes = []
    for det in results.boxes:

        # Extract bounding box info
        class_id = float(det.cls)
        class_name = results.names[class_id]
        confidence = float(det.conf)
        xyxy = det.xyxy[0].tolist()  # [xmin, ymin, xmax, ymax]

        boxes.append({
            "class": class_id,
            "confidence": confidence,
            "xmin": xyxy[0],
            "ymin": xyxy[1],
            "xmax": xyxy[2],
            "ymax": xyxy[3]
        })
        print("this ran")

    return boxes


"""
    TCP NETWORKING SUPPORT METHODS
"""


def initialize_server():
    try:
        print("Initializing TCP server")
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((TCP_IP, TCP_PORT))
        server_socket.listen(1)
        print(f"Listening on {TCP_IP}:{TCP_PORT}")
        return server_socket
    except Exception as e:
        print(f"Failed to initialize server on port {TCP_PORT}: {e}")
        sys.exit(1)


def execute_predictive_control(server_socket):
    while True:
        try:
            conn, address = server_socket.accept()
            print(f"Connected to Unity client at {address} for model {MODEL_IDX}")

            try:
                while True:
                    # Read image size (4 bytes)
                    size_bytes = conn.recv(4)
                    if not size_bytes:
                        print("Connection closed by Unity")
                        break
                    img_size = int.from_bytes(size_bytes, byteorder='little')

                    # Read image data
                    img_data = b''
                    while len(img_data) < img_size:
                        data = conn.recv(min(BUFFER_SIZE, img_size - len(img_data)))
                        if not data:
                            print("Connection closed by Unity during image read")
                            break
                        img_data += data
                    print(f"Received image of {len(img_data)} bytes from {address}")

                    converted_img = string_to_image(img_data)
                    resized_img = resize(converted_img)
                    boxes = detect_objects(resized_img, model)

                    # Send control response
                    control_message = json.dumps(boxes).encode('utf-8')
                    conn.send(len(control_message).to_bytes(4, byteorder='little'))
                    conn.send(control_message)
                    print(f"Sent controls for model {MODEL_IDX}: {boxes}")
            except Exception as e:
                print(f"Error in predictive control loop: {e}")
            finally:
                conn.close()
                print("Connection closed")
        except Exception as e:
            print(f"Error accepting connection: {e}")
            break


"""
    NETWORK / DATA / ML VERIFICATION METHODS
"""


def verify_byte_str_received(data):
    print("Verifying the data received from Unity")
    sent_checksum = hashlib.md5(data).hexdigest()
    print(f"Received checksum: {sent_checksum}")
    # img = tf.io.decode_image(data, channels=3, dtype=tf.float32) # use your decode and process
    # print(f"Image shape: {img.shape}, dtype: {img.dtype}")
    # plt.imshow(img)
    # plt.show()
    # print()

def verify_tcp_messaging(server_socket):
    try:
        conn, address = server_socket.accept()
        print(f"Connected to Unity client at {address}")

        try:
            while True:
                size_bytes = conn.recv(4)
                if not size_bytes:
                    print("Connection closed by Unity")
                    break
                img_size = int.from_bytes(size_bytes, byteorder='little')

                img_data = b''
                while len(img_data) < img_size:
                    data = conn.recv(min(BUFFER_SIZE, img_size - len(img_data)))
                    if not data:
                        print("Connection closed by Unity")
                        break
                    img_data += data
                print(f"Received image of {len(img_data)} bytes")
                verify_byte_str_received(img_data)
        finally:
            conn.close()
            print("Connection closed")
    except Exception as e:
        print(f"Error in verification loop: {e}")


"""
    MAIN EXECUTION
"""

if __name__ == '__main__':
    # Load the
    load_model()

    img_path = 'C:/Users/alexah1/Documents/GitHub/ML-Unity-Project/MCA-ML/data/zoo.jpg'
    # img = resize(img_path)
    img = cv2.imread(img_path, 3)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxesJson = detect_objects(img, model)


    # Initialize the TCP server
    server_socket = initialize_server()

    try:
        # Test TCP service
        if DEBUG_FLAG:
            verify_tcp_messaging(server_socket)
        else:
            execute_predictive_control(server_socket)
    except KeyboardInterrupt:
        print("Shutting down server")
    finally:
        server_socket.close()
        print(f"Closed TCP server on port {TCP_PORT}")