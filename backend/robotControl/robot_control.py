# backend/robotControl/robot_control.py
from unitree_sdk2py.go2.sport.sport_client import SportClient

client = SportClient()
client.SetTimeout(10.0)
client.Init()


def execute_action(action: str):
    # --- Movement ---
    if action == "move_forward":
        client.Move(0.5, 0, 0)
    elif action == "move_backward":
        client.Move(-0.3, 0, 0)
    elif action == "move_left":
        client.Move(0, 0.3, 0)
    elif action == "move_right":
        client.Move(0, -0.3, 0)
    elif action == "turn_left":
        client.Move(0, 0, 0.3)
    elif action == "turn_right":
        client.Move(0, 0, -0.3)
    elif action == "stop":
        client.StopMove()

    # --- Posture ---
    elif action == "stand_up":
        client.StandUp()
    elif action == "stand_down":
        client.StandDown()
    elif action == "balance_stand":
        client.BalanceStand()
    elif action == "sit":
        client.Sit()
    elif action == "rise_sit":
        client.RiseSit()
    elif action == "recovery_stand":
        client.RecoveryStand()
    elif action == "damp":
        client.Damp()

    # --- Emotes / Expressions ---
    elif action == "hello" or action == "emote_wave":
        client.Hello()
    elif action == "dance1" or action == "emote_dance":
        client.Dance1()
    elif action == "dance2":
        client.Dance2()
    elif action == "stretch":
        client.Stretch()
    elif action == "pose":
        client.Pose(True)
    elif action == "heart":
        client.Heart()
    elif action == "scrape":
        client.Scrape()
    elif action == "content":
        client.Content()

    # --- Acrobatics (use with caution) ---
    elif action == "front_flip":
        client.FrontFlip()
    elif action == "front_jump":
        client.FrontJump()
    elif action == "front_pounce":
        client.FrontPounce()
    elif action == "left_flip":
        client.LeftFlip()
    elif action == "back_flip":
        client.BackFlip()

    # --- Gait Modes ---
    elif action == "static_walk":
        client.StaticWalk()
    elif action == "trot_run":
        client.TrotRun()
    elif action == "free_walk":
        client.FreeWalk()
    elif action == "classic_walk_on":
        client.ClassicWalk(True)
    elif action == "classic_walk_off":
        client.ClassicWalk(False)
    elif action == "walk_upright_on":
        client.WalkUpright(True)
    elif action == "walk_upright_off":
        client.WalkUpright(False)
    elif action == "cross_step_on":
        client.CrossStep(True)
    elif action == "cross_step_off":
        client.CrossStep(False)
    elif action == "free_bound_on":
        client.FreeBound(True)
    elif action == "free_bound_off":
        client.FreeBound(False)
    elif action == "free_jump_on":
        client.FreeJump(True)
    elif action == "free_jump_off":
        client.FreeJump(False)
    elif action == "free_avoid_on":
        client.FreeAvoid(True)
    elif action == "free_avoid_off":
        client.FreeAvoid(False)
    elif action == "hand_stand_on":
        client.HandStand(True)
    elif action == "hand_stand_off":
        client.HandStand(False)

    # --- Speed ---
    elif action == "speed_slow":
        client.SpeedLevel(1)
    elif action == "speed_normal":
        client.SpeedLevel(2)
    elif action == "speed_fast":
        client.SpeedLevel(3)

    # --- Safe default ---
    else:
        print(f"[RobotControl] Unknown action: '{action}' — stopping")
        client.StopMove()