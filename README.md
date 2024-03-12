# Mitigating SYN Floods: An AI-Driven Approach to DDoS Attacks Detection and Prevention in Real-Time

## What is SYN flood?
A SYN flood is a type of cyber attack that targets the three-way handshake process used in establishing a TCP connection between two devices on a network (e.g., a device and a server). In a typical TCP handshake, the client sends a SYN (synchronize) packet to the server, the server responds with a SYN-ACK (synchronize-acknowledge) packet, and finally, the client acknowledges the server's response with an ACK packet. This process establishes a connection.

In a SYN flood attack, the attacker floods the target server with a large number of SYN packets, typically with spoofed source IP addresses or by simply not responding to the SYN-ACK packets sent by the server. The server then waits for ACK packets that never arrive, tying up resources as it holds open incomplete connections. This can exhaust the server's resources such as memory or processing power, causing legitimate connection requests to be denied or severely slowed down.

## How it can be mitigated?
SYN flood attacks can be mitigated using techniques such as SYN cookies, rate limiting, and specialized hardware or software solutions designed to identify and block malicious traffic. Those methods and techniques have been suggested by CISCO RFC 4987 which can be accessed [here](https://datatracker.ietf.org/doc/rfc4987/)

## What is our approach?
In this repository, I'm aiming to detect and prevent SYN floods in real-time using AI techniques such as Gradient Boosting, Extreme Gradient Boosting, and Neural Networks.

## How does it work?
### 1. Dataset:
To lay the groundwork for our models, we selected a dataset that encapsulates the characteristics of SYN flood attacks, facilitating the training of our models on relevant data prior to real-time evaluation. The chosen dataset for this purpose is CIC-DDOS2019, accessible through the Canadian Institute for Cybersecurity's website. This dataset is particularly comprehensive, comprising over one million data points and eighty-five distinct features, enabling the effective differentiation between SYN flood attacks and regular network traffic, which can be found [here](https://www.unb.ca/cic/about/index.html)

Given the computational limitations encountered, the dataset was condensed to a more manageable size of 10,000 data points. This subset, while significantly smaller, retains sufficient variability and complexity to train our models effectively, ensuring they are well-prepared for subsequent real-time testing scenarios.

### 2. Feature Selection:
Given the objective of this project to identify SYN flood attacks in real-time settings, it is imperative that our models demonstrate both rapid responsiveness and efficacy. To this end, we have carefully selected a set of features that underpin these requirements. Below is a concise list of the chosen features that will be utilized to train our models:

- **‘Source Port’**: In a SYN flood attack, source ports may be random or spoofed, unlike in normal communications where source ports are usually consistent for a given application.
- **‘Destination Port’**: During a SYN flood, the destination port is often a well-known service port (e.g., 80 for HTTP, 443 for HTTPS) targeted to exhaust resources. Normal communications have more varied destination ports based on the services being used.
- **‘Protocol’**: The protocol is usually TCP in both SYN flood attacks and normal communications, as SYN packets are part of the TCP protocol.
- **‘Flow Duration’**: In SYN flood attacks, flow durations tend to be very short, as attackers only initiate connections without completing them. Normal communications have longer flow durations due to the exchange of data.
- **‘Total Fwd Packets’**: In a SYN flood, the total forward packets (initial packets from the source to the destination) are usually high with mostly SYN packets, whereas normal communications have a more balanced exchange of packets.
- **‘Total Backward Packets’**: During a SYN flood attack, the total backward packets (responses from the destination to the source) are significantly lower compared to normal traffic, due to the lack of established connections.
- **‘Flow Bytes/s’**: The flow bytes per second can be significantly lower in SYN flood attacks since only initial SYN packets are sent without data payloads. Normal communications involve data transfer, leading to higher flow bytes per second.
- **‘Flow Packets/s’**: The flow packets per second could be much higher in SYN flood scenarios due to the rapid sending of SYN packets to overwhelm the target. In contrast, normal communications typically have a more moderate and consistent packet rate.

### 3. Testing platform:
Following the selection of essential features, the next step involves real-time computation of these features to enable their integration into our models for prediction purposes. To achieve this, a testing platform was developed, capable of analyzing network packets in real-time. Utilizing "pyshark," a Python wrapper for tshark, we efficiently captured raw data packets and transformed them into structured features comprehensible by our models. This transformation process facilitated the models' ability to make informed predictions based on the processed data. The methodology employed for calculating these features is detailed below:
- **‘Source Port’**: This feature is derived directly from the TCP layer of each packet encountered in the network capture. The source port, which is a numerical identifier for the sending process, is extracted from the TCP header information.
- **‘Destination Port’**: Like the source port, the destination port is also extracted from the TCP layer of each packet.
- **‘Protocol’**: The protocol used for the communication is indicated by a numerical value. In this context, the value is set to “6”, which corresponds to TCP (Transmission Control Protocol) in the protocol suite used by the IP (Internet Protocol).
- **‘Flow Duration’**: The flow duration represents the total time elapsed from the beginning of the data flow until the current packet. It is calculated by taking the current timestamp (the time when the current packet is processed) and subtracting the timestamp of when the first packet was processed (start_time).
- **‘Total Fwd Packets’**: This count represents the number of packets sent from a specific source IP address, which is predefined (in this case, 192.168.10.245). Each time a packet with this source IP is encountered, the counter is incremented.
- **‘Total Backward Packets’**: This feature counts total packets that are not originating from the predefined source IP.
- **‘Flow Bytes/s’**: This feature calculates the rate at which data is transmitted over the network flow. It is calculated by dividing the total bytes (the sum of the lengths of all packets processed so far) by the flow duration.
- **‘Flow Packets/s’**: Like the ‘flow bytes per second’, this feature measures the rate of packets transmitted in the flow. It is calculated by adding the total forward and backward packets and then dividing by the flow duration.

### 4. Model Developments:
In this section, our primary emphasis is on developing our models and assessing their performance using testing data. This process is crucial to verify that the models are prepared and effective for real-time evaluation scenarios such as Confusion matrix, Accuracy, Recall, precision, F1 score.

### 5. Model Deployment:
With our testing platform and models primed, the next step involves integrating the models into the platform to enable real-time predictions. Upon completion of their development and training, each model, along with its corresponding scaler, has been preserved for operational use within the testing environment. The integration process simply requires importing the pre-trained models and scalers into the testing platform, allowing it to commence predictive analysis.

Additionally, the platform incorporates several features designed to enhance its functionality and user experience. One such feature is the capability to monitor the computational resources utilized by the model during prediction processes, ensuring efficient resource management. Another significant feature is the implementation of an IP blocking mechanism. This mechanism is triggered when an IP address is identified as a source of SYN flood attacks, leading to its temporary blockade within the system's firewall. Users are informed of this action and provided with a timer; the blockade is lifted upon the timer's expiration, facilitating further investigation of the implicated IP address. This approach not only strengthens the system's defensive measures but also empowers users to conduct in-depth analyses of potential security threats.

### 6. Testing:
The testing methodology for this project was conducted using two virtual machines: one operating Windows 10, serving as the victim, and the other running Kali Linux, acting as the attacker. This setup allowed us to create a controlled environment for our experiments. For the execution of SYN flood attacks, we utilized the MSF console method, targeting the victim device. This approach enabled us to assess the real-time efficacy of our models in detecting and responding to such cybersecurity threats.



