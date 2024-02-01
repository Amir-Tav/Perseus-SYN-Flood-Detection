import tkinter as tk
from tkinter import ttk
import pyshark
import pandas as pd
import time
import joblib  # For loading joblib models (GB-model, XGB-model)

import tensorflow as tf  # For loading and using the TensorFlow model (NN-model)

# Set the path to TShark
pyshark.tshark.tshark.TSHARK_PATH = 'C:\\Program Files\\Wireshark'

scaler = joblib.load('scaler.save')
Perseus = joblib.load('GB-model')
Perseus = joblib.load('XGB-model')

def main():
    # Model selection prompt
    model_choice = input("Select a model (GB for Gradient Boosting, XGB for Extreme Gradient Boosting): ").strip().lower()
    if model_choice == 'gb':
        model_path = 'GB-model'
    elif model_choice == 'xgb':
        model_path = 'XGB-model'
    else:
        print("Invalid model selection. Please choose either 'GB' or 'XGB'.")
        return

    # Load the selected model and the scaler
    model = joblib.load(model_path)
    scaler = joblib.load('scaler.save')

    # Capture network packets from the specified interface
    capture = pyshark.LiveCapture(interface='\\Device\\NPF_{D26B2253-EE3F-4BC2-8451-8CE0C1E16A7D}', display_filter='tcp.port==443')

    # Initialize counters and start time
    total_fwd_packets = 0
    total_bwd_packets = 0
    start_time = time.time()
    total_bytes = 0
    
    for packet in capture.sniff_continuously():
        # Packet processing logic remains the same as in your original code
        # Check if the packet has a TCP layer
        if 'TCP' in packet:
            tcp_layer = packet.tcp
            ip_layer = packet.ip

            # Extract Source and Destination Ports
            source_port = int(tcp_layer.srcport)
            destination_port = int(tcp_layer.dstport)

            #Extracting source and destination IP's
            source_ip = ip_layer.src
            destination_ip = ip_layer.dst

            # Protocol 
            protocol = 6  # TCP protocol number

            # Count forward and backward packets
            if ip_layer.src == '192.168.10.245':
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

            # Print the calculated features
            features_string = ", ".join(f"\033[1m{key}\033[0m: {value}" for key, value in features_df.iloc[0].to_dict().items())
            print(f"{features_string}")

            # Check for attack
            if prediction[0] == 1:  # 1 denotes attack
                 print(f"SYN flood attack detected! Source IP: {source_ip}, Destination IP: {destination_ip}")

                 break# Stop capturing packets once an attack is detected


if __name__ == "__main__":
    main()
