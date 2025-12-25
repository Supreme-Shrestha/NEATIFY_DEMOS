"""
Distributed Training Protocol for Self-Driving Car
Uses JSON over TCP sockets for communication between master and workers.
"""
import json
import pickle
import base64

class Message:
    """Message types for distributed communication."""
    REGISTER = "REGISTER"
    TASK = "TASK"
    RESULT = "RESULT"
    SHUTDOWN = "SHUTDOWN"
    HEARTBEAT = "HEARTBEAT"

def serialize_genome(genome):
    """Serialize genome to base64 string."""
    genome_bytes = pickle.dumps(genome)
    return base64.b64encode(genome_bytes).decode('utf-8')

def deserialize_genome(genome_str):
    """Deserialize genome from base64 string."""
    genome_bytes = base64.b64decode(genome_str.encode('utf-8'))
    return pickle.loads(genome_bytes)

def create_message(msg_type, data=None):
    """Create a JSON message."""
    return json.dumps({
        'type': msg_type,
        'data': data or {}
    })

def parse_message(msg_str):
    """Parse a JSON message."""
    return json.loads(msg_str)

def send_message(sock, msg_type, data=None):
    """Send a message over socket."""
    msg = create_message(msg_type, data)
    msg_bytes = msg.encode('utf-8')
    # Send length prefix
    length = len(msg_bytes)
    sock.sendall(length.to_bytes(4, 'big'))
    sock.sendall(msg_bytes)

def receive_message(sock):
    """Receive a message from socket."""
    # Receive length prefix
    length_bytes = sock.recv(4)
    if not length_bytes:
        return None
    length = int.from_bytes(length_bytes, 'big')
    
    # Receive message
    msg_bytes = b''
    while len(msg_bytes) < length:
        chunk = sock.recv(min(length - len(msg_bytes), 4096))
        if not chunk:
            return None
        msg_bytes += chunk
    
    msg_str = msg_bytes.decode('utf-8')
    return parse_message(msg_str)
