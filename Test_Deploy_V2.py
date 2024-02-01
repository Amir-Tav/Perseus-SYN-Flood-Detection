import pyshark
import time
import joblib
import time
import pandas as pd
import tkinter as tk
import subprocess
from threading import Thread
import tensorflow as tf

# Set the path to TShark
pyshark.tshark.tshark.TSHARK_PATH = 'C:\\Program Files\\Wireshark'

# Load the trained model and the scaler
# Perseus = joblib.load('GB-model')
Perseus = tf.keras.models.load_model('NN-model')
scaler = joblib.load('scaler.save')

blocking_duration = 10

def update_popup_message(window, message_label, message):
    # Update the message in the popup window
    message_label.config(text=message)
    window.update()

def update_timer(window, timer_label, extend_button, start_time):
    global blocking_duration
    while time.time() - start_time < blocking_duration:
        remaining_time = max(blocking_duration - int(time.time() - start_time), 0)
        timer_label.config(text=f"Time remaining: {remaining_time} seconds")
        window.update()
        time.sleep(1)

    # Remove the timer label and the extend button once blocking is done
    timer_label.pack_forget()
    extend_button.pack_forget()

def block_ip(window, message_label, timer_label, extend_button, ip_address):
    global blocking_duration
    # Block the IP
    start_time = time.time()
    update_popup_message(window, message_label, f"SYN Flood has been detected at: {ip_address}. Blocking IP: {ip_address} for: ")
    subprocess.call(f"netsh advfirewall firewall add rule name=\"block{ip_address}\" dir=in interface=any action=block remoteip={ip_address}", shell=True)
    
    # Start the timer
    Thread(target=update_timer, args=(window, timer_label, extend_button, start_time)).start()

    # Wait for the duration
    while time.time() - start_time < blocking_duration:
        time.sleep(1)  # Sleep in short intervals for responsiveness
    
    # Unblock the IP
    subprocess.call(f"netsh advfirewall firewall delete rule name=\"block{ip_address}\"", shell=True)
    update_popup_message(window, message_label, f"IP: {ip_address} has been unblocked from fire wall rules.")

def extend_blocking_duration():
    global blocking_duration
    blocking_duration += 10

def show_popup(ip_address):
    # Create a popup window
    window = tk.Tk()
    window.title("Alert")
    window.geometry("450x250")  # Set initial size of the window

    # Set the theme colors
    background_color = "#263D42"
    text_color = "#FFFFFF"
    button_color = "#1E6262"

    window.configure(background=background_color)
    
    # Attack message label
    message_label = tk.Label(window, padx=20, pady=20, bg=background_color, fg=text_color)
    message_label.pack()

    # Timer label
    timer_label = tk.Label(window, text="Time remaining: 40 seconds", padx=20, bg=background_color, fg=text_color)
    timer_label.pack()

    # Button to extend the blocking duration
    extend_button = tk.Button(window, text="Extend Blocking by 10s", command=extend_blocking_duration, bg=button_color, fg=text_color)
    extend_button.pack(pady=10)  # Add some space around the button

    # Start blocking IP in a separate thread
    Thread(target=block_ip, args=(window, message_label, timer_label, extend_button, ip_address)).start()

    window.mainloop()



def main():
    
    # Capture network packets from the specified interface
    capture = pyshark.LiveCapture(interface='\\Device\\NPF_{D26B2253-EE3F-4BC2-8451-8CE0C1E16A7D}',display_filter= 'tcp.port==443')

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
            if prediction[0][0] == 0.7:  # Assuming 1 denotes an attack
                print(f"Attack detected! Source IP: {source_ip}, Destination IP: {destination_ip}")
                # Start a thread to handle the alert and potentially block the IP
                Thread(target=show_popup, args=(destination_ip,)).start()

                break  # Optionally, stop capturing packets once an attack is detected


if __name__ == "__main__":
    main()

