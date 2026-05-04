import asyncio
import sys

import pygame
from bleak import BleakScanner, BleakClient


# --- CONFIGURATION ---
UART_CHARACTERISTIC_UUID = "0000ffe3-0000-1000-8000-00805f9b34fb"

# Robot Constraints
MAX_LINEAR_VELOCITY = 200.0  # [mm/s]
MAX_ANGULAR_VELOCITY = 0.75  # [rad/s]
DEADZONE = 0.08  # Joystick deadzone


def apply_curve(val, deadzone):
    if abs(val) < deadzone:
        return 0.0

    # Scale the remaining travel from 0.0 to 1.0
    sign = 1.0 if val > 0 else -1.0
    scaled_val = (abs(val) - deadzone) / (1.0 - deadzone)

    # Cube the output (x^3). This gives extreme precision near the center,
    # but still allows maximum speed when pushed all the way to the edge.
    return sign * (scaled_val ** 3)


async def main():
    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        print("Error: No gamepad connected. Please connect your PS5 controller.")
        sys.exit(1)

    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"🎮 Connected to Gamepad: {joystick.get_name()}")

    print("🔍 Scanning for Bluetooth devices...")
    devices = await BleakScanner.discover(timeout=5.0)

    robot_device = None
    for d in devices:
        if d.name and ("Makeblock" in d.name or "ELE" in d.name):
            robot_device = d
            break

    if not robot_device:
        print("❌ Could not find the Makeblock robot.")
        sys.exit(1)

    print(f"✅ Found Robot: {robot_device.name} [{robot_device.address}]")

    print("🔗 Connecting...")
    try:
        async with BleakClient(robot_device) as client:
            print("🚀 Connected! Use the sticks to drive. Press Ctrl+C to stop.")

            import time  # Ensure this is at the top of your script

            # Control loop setup
            toggle_space = False
            last_send_time = time.time()
            last_v = -999.0
            last_w = -999.0

            packet_counter = 0  # <-- NEW: Packet counter

            while True:
                pygame.event.pump()

                raw_y = joystick.get_axis(1)
                raw_x = joystick.get_axis(2)

                forward_input = apply_curve(-raw_y, DEADZONE)
                turn_input = apply_curve(-raw_x, DEADZONE)

                v = forward_input * MAX_LINEAR_VELOCITY
                w = turn_input * MAX_ANGULAR_VELOCITY

                packet_counter = (packet_counter + 1) % 10

                current_time = time.time()

                # Send if the joystick moved, OR if 0.25 seconds have passed (Heartbeat)
                if abs(v - last_v) > 0.5 or abs(w - last_w) > 0.05 or (current_time - last_send_time) > 0.25:

                    # Toggle a space at the end to completely defeat OS Bluetooth deduplication
                    toggle_space = not toggle_space
                    space = " " if toggle_space else ""

                    command = f"V{v:.2f}W{w:.2f}{space}\n"

                    try:
                        await client.write_gatt_char(UART_CHARACTERISTIC_UUID, command.encode('utf-8'), response=False)
                    except Exception:
                        pass

                    last_v = v
                    last_w = w
                    last_send_time = current_time

                # Run loop at 50Hz for ultra-low latency stick reading
                await asyncio.sleep(0.02)

    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"❌ Disconnected or error: {e}")
    finally:
        pygame.quit()
        print("🛑 Stopped.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nUser requested exit.")
