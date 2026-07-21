from unitree_sdk2py.go2.sport.sport_client import SportClient

client = SportClient()
client.SetTimeout(10.0)
client.Init()

def execute_action(action: str):
    if action == "follow_person":
        client.Move(0.3, 0, 0)      # move forward slowly
    elif action == "move_forward":
        client.Move(0.5, 0, 0)
    elif action == "move_backward":
        client.Move(-0.3, 0, 0)
    elif action == "turn_left":
        client.Move(0, 0, 0.5)
    elif action == "turn_right":
        client.Move(0, 0, -0.5)
    elif action == "stop":
        client.StopMove()
    elif action == "avoid_obstacle":
        client.Move(0, 0.3, 0)      # strafe sideways
    else:
        client.StopMove()           # safe default