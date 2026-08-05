# Import the communication initializer from the Unitree SDK
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

# Import the Go2 SportClient
# SportClient contains high-level motion methods
from unitree_sdk2py.go2.sport.sport_client import SportClient

# Used for delays between robot actions
import time

# Used to optionally pass robot IP from terminal arguments
import sys


def main():

    # ---------------------------------------------------
    # INITIALIZE COMMUNICATION
    # ---------------------------------------------------

    # If an IP address is passed into terminal:
    # python3 move_test.py 192.168.123.161
    #
    # then initialize communication using that IP

    if len(sys.argv) > 1:

        # Initialize DDS communication with robot using IP
        ChannelFactoryInitialize(0, sys.argv[1])

    else:

        # Initialize communication automatically
        # Works when running directly onboard robot
        ChannelFactoryInitialize(0)

    print("Communication initialized")


    # ---------------------------------------------------
    # CREATE SPORT CLIENT
    # ---------------------------------------------------

    # SportClient provides:
    # - movement
    # - posture
    # - gait
    # - balance
    # - locomotion controls

    client = SportClient()

    # Set timeout duration for SDK requests
    # If robot does not respond within 10 seconds,
    # the request will fail
    client.SetTimeout(10.0)

    # Initialize the client
    client.Init()

    print("Sport client initialized")


    # ---------------------------------------------------
    # STAND UP
    # ---------------------------------------------------

    print("Standing up...")

    # Makes robot stand up into active state
    client.StandUp()

    # Wait for robot to finish motion
    time.sleep(3)


    # ---------------------------------------------------
    # MOVE FORWARD
    # ---------------------------------------------------

    print("Moving forward...")

    # Move(vx, vy, vyaw)
    #
    # vx   = forward/backward velocity
    # vy   = left/right velocity
    # vyaw = rotational velocity
    #
    # Positive vx = forward
    # Negative vx = backward

    client.Move(0.3, 0.0, 0.0)

    # Keep moving for 3 seconds
    time.sleep(3)

    # Stop robot movement
    client.StopMove()

    time.sleep(1)


    # ---------------------------------------------------
    # MOVE BACKWARD
    # ---------------------------------------------------

    print("Moving backward...")

    client.Move(-0.2, 0.0, 0.0)

    time.sleep(3)

    client.StopMove()

    time.sleep(1)


    # ---------------------------------------------------
    # STRAFE LEFT
    # ---------------------------------------------------

    print("Moving left...")

    # Positive vy = move left
    client.Move(0.0, 0.2, 0.0)

    time.sleep(3)

    client.StopMove()

    time.sleep(1)


    # ---------------------------------------------------
    # STRAFE RIGHT
    # ---------------------------------------------------

    print("Moving right...")

    # Negative vy = move right
    client.Move(0.0, -0.2, 0.0)

    time.sleep(3)

    client.StopMove()

    time.sleep(1)


    # ---------------------------------------------------
    # ROTATE LEFT
    # ---------------------------------------------------

    print("Rotating left...")

    # Positive vyaw = rotate left
    client.Move(0.0, 0.0, 0.5)

    time.sleep(3)

    client.StopMove()

    time.sleep(1)


    # ---------------------------------------------------
    # ROTATE RIGHT
    # ---------------------------------------------------

    print("Rotating right...")

    # Negative vyaw = rotate right
    client.Move(0.0, 0.0, -0.5)

    time.sleep(3)

    client.StopMove()

    time.sleep(1)

    # ---------------------------------------------------
    # SIT / STAND DOWN
    # ---------------------------------------------------

    print("Standing down...")

    # Robot returns to resting/sitting posture
    client.StandDown()

    time.sleep(3)


    print("Demo complete")


# ---------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------

# Runs main() when script starts
if __name__ == "__main__":
    main()
