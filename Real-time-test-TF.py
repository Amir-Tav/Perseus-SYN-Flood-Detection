import pyshark
import time
import joblib
import pandas as pd
import tensorflow as tf  # Import TensorFlow

# Load the trained model and the scaler
scaler = joblib.load('scaler.save')
Perseus = tf.keras.models.load_model('NN-model')  # Load the TensorFlow model

def main():
    # Set the path to TShark
    pyshark.tshark.tshark.TSHARK_PATH = 'C:\\Program Files\\Wireshark'
    
    # Capture network packets from the specified interface
    capture = pyshark.LiveCapture(interface='\\Device\\NPF_{D26B2253-EE3F-4BC2-8451-8CE0C1E16A7D}', display_filter='tcp.port==443')

    # Initialize counters and start time
    total_fwd_packets = 0
    total_bwd_packets = 0
    start_time = time.time()
    total_bytes = 0

    for packet in capture.sniff_continuously():
        # Check if the packet has a TCP layer
        if 'TCP' in packet:
            tcp_layer = packet.tcp
            ip_layer = packet.ip

            # Extract Source and Destination Ports
            source_port = int(tcp_layer.srcport)
            destination_port = int(tcp_layer.dstport)

            # Extracting source and destination IP's
            source_ip = ip_layer.src
            destination_ip = ip_layer.dst

            # Protocol
            protocol = 6  # TCP protocol number

            # Count forward and backward packets
            if ip_layer.src == '192.168.10.245':  # Replace with your network's IP if needed
                total_fwd_packets += 1
            else:
                total_bwd_packets += 1

            # Accumulate total bytes
            total_bytes += int(packet.length)

            # Calculate flow duration
            current_time = time.time()
            flow_duration = current_time - start_time

            # Calculate Flow Bytes/s and Flow Packets/s
            flow_bytes_per_second = total_bytes / flow_duration if flow_duration > 0 else 0
            flow_packets_per_second = (total_fwd_packets + total_bwd_packets) / flow_duration if flow_duration > 0 else 0

            # Create a DataFrame for the features
            feature_names = [
                'Source Port', 'Destination Port', 'Protocol', 'Flow Duration',
                'Total Fwd Packets', 'Total Backward Packets', 'Flow Bytes/s', 'Flow Packets/s'
            ]
            feature_values = [
                source_port, destination_port, protocol, flow_duration,
                total_fwd_packets, total_bwd_packets, flow_bytes_per_second, flow_packets_per_second
            ]
            features_df = pd.DataFrame([feature_values], columns=feature_names)

            # Scale the features
            features_scaled = scaler.transform(features_df)

            # Make a prediction
            prediction = Perseus.predict(features_scaled)
            predicted_class = (prediction > 0.5).astype(int)  # Convert probabilities to binary class labels

            # Print the calculated features
            features_string = ", ".join(f"{key}: {value}" for key, value in features_df.iloc[0].to_dict().items())
            print(f"Features: {features_string}")

            # Check for attack
            if predicted_class[0][0] == 1:  # 1 denotes attack
                end_time = time.time()  # Stop the timer when the prediction is satisfied
                elapsed_time = end_time - start_time  # Calculate total elapsed time
                print(f"Total time until detection: {elapsed_time:.2f} seconds")  # Display total elapsed time
                print(f"SYN flood attack detected! Source IP: {source_ip}, Destination IP: {destination_ip}")
                break  # Stop capturing packets once an attack is detected

if __name__ == "__main__":
    main()
