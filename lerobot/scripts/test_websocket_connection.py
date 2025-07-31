#!/usr/bin/env python
"""Test script to diagnose WebSocket connection issues."""

import argparse
import socket
import sys
import time
from urllib.parse import urlparse

import websockets.sync.client


def test_port_availability(host: str, port: int) -> bool:
    """Test if a port is available for connection."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception as e:
        print(f"❌ Error testing port {port} on {host}: {e}")
        return False


def test_websocket_connection(uri: str) -> bool:
    """Test WebSocket connection to the given URI."""
    print(f"🔗 Testing WebSocket connection to: {uri}")
    
    try:
        # Parse URI
        parsed = urlparse(uri)
        if not parsed.scheme:
            uri = f"ws://{uri}"
            parsed = urlparse(uri)
        
        host = parsed.hostname
        port = parsed.port or 8001
        
        print(f"📍 Host: {host}")
        print(f"🔌 Port: {port}")
        
        # Test basic TCP connection first
        print(f"🔍 Testing TCP connection to {host}:{port}...")
        if test_port_availability(host, port):
            print(f"✅ TCP connection to {host}:{port} is available")
        else:
            print(f"❌ TCP connection to {host}:{port} is not available")
            print(f"💡 Make sure the server is running and the port is not blocked by firewall")
            return False
        
        # Test WebSocket connection
        print(f"🔗 Attempting WebSocket connection...")
        conn = websockets.sync.client.connect(
            uri,
            compression=None,
            max_size=None,
            ping_interval=20,
            ping_timeout=20,
            close_timeout=10,
        )
        
        print(f"✅ WebSocket connection successful!")
        
        # Try to receive metadata
        try:
            metadata = conn.recv()
            print(f"📥 Received metadata from server")
            print(f"📋 Metadata: {metadata[:100]}..." if len(str(metadata)) > 100 else f"📋 Metadata: {metadata}")
        except Exception as e:
            print(f"⚠️  Could not receive metadata: {e}")
        
        conn.close()
        return True
        
    except websockets.exceptions.ConnectionRefusedError:
        print(f"❌ Connection refused - server may not be running")
        print(f"💡 Start the server with: python lerobot/scripts/serve_widowx.py")
        return False
    except websockets.exceptions.InvalidURI as e:
        print(f"❌ Invalid URI: {e}")
        return False
    except websockets.exceptions.InvalidHandshake as e:
        print(f"❌ WebSocket handshake failed: {e}")
        print(f"💡 This might indicate the server is not a WebSocket server")
        return False
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test WebSocket connection to policy server")
    parser.add_argument(
        "--server-uri",
        type=str,
        default="ws://localhost:8001",
        help="WebSocket server URI to test (e.g., ws://localhost:8001)"
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of connection attempts"
    )
    
    args = parser.parse_args()
    
    print(f"🧪 WebSocket Connection Test")
    print(f"=" * 50)
    
    success = False
    for attempt in range(1, args.retries + 1):
        print(f"\n🔄 Attempt {attempt}/{args.retries}")
        print(f"-" * 30)
        
        if test_websocket_connection(args.server_uri):
            success = True
            print(f"\n✅ Connection test successful!")
            break
        else:
            print(f"\n❌ Connection test failed (attempt {attempt})")
            if attempt < args.retries:
                print(f"⏳ Waiting 2 seconds before retry...")
                time.sleep(2)
    
    if not success:
        print(f"\n❌ All connection attempts failed")
        print(f"\n🔧 Troubleshooting tips:")
        print(f"1. Make sure the server is running: python lerobot/scripts/serve_widowx.py")
        print(f"2. Check if the port is in use: lsof -i :8001")
        print(f"3. Check firewall settings")
        print(f"4. Try a different port: --port 8002")
        sys.exit(1)
    else:
        print(f"\n🎉 Connection test completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main() 