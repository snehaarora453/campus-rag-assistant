# Networking Notes

## TCP vs UDP

**TCP (Transmission Control Protocol)**
- Connection-oriented: requires a handshake before data transfer.
- Reliable: guarantees delivery and correct ordering via acknowledgments and retransmission.
- Slower due to the overhead of reliability guarantees.
- Used for: web browsing (HTTP), file transfer, email.

**UDP (User Datagram Protocol)**
- Connectionless: no handshake, just sends packets.
- Unreliable: no guarantee of delivery or order.
- Faster and lower overhead than TCP.
- Used for: video streaming, online gaming, DNS lookups.

## The OSI Model (7 layers)
1. Physical
2. Data Link
3. Network
4. Transport
5. Session
6. Presentation
7. Application
