#!/usr/bin/env python3
"""CLI for jogging the arm.

Two ways to use it:

1. One-shot: each invocation is its own process; the arm's last commanded
   position is persisted to a small JSON state file (arm_state.json, next
   to this script) and reloaded on the next call. Good for driving from
   another program (e.g. one CLI call per recognized voice command).

       python3 jog_arm.py forward --send
       python3 jog_arm.py claw-open --send

2. Interactive: start it once, then type commands at a prompt without
   relaunching. State lives in memory (and is still saved to disk on every
   command, so one-shot calls later pick up where you left off). This is
   also the easier shape to wire a speech recognizer into later -- feed
   recognized text into this same running process instead of spawning a
   new one per phrase.

       python3 jog_arm.py interactive --send
       > forward
       > forward 20
       > claw open
       > back
       > claw closed
       > quit

Motion reuses the same trajectory/validation machinery as
pick_and_place.py, so every jog is smoothed and speed/step limited before
anything is sent.

Drop this file in the same directory as robot_inverse_kinematics.py and
pick_and_place.py.

Voice integration later just needs to map a transcribed phrase to one of
these subcommands, e.g. "arm forward" -> "forward".
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from robot_inverse_kinematics import (
    IKError,
    SERVO_ZERO_DEG,
    control_to_joint,
    controls_within_limits,
    forward_kinematics,
    DEFAULT_GEOMETRY,
)
import pick_and_place as pp


STATE_PATH = Path(__file__).resolve().parent / "arm_state.json"

DEFAULT_STEP_MM = 10.0

# Home position used only when no state file exists yet (e.g. first run,
# or after `reset`). Matches the servo-zero pose from
# robot_inverse_kinematics.py with the gripper open.
HOME_CONTROL_DEG = np.concatenate(
    (SERVO_ZERO_DEG, [pp.DEFAULT_GRIPPER_OPEN_DEG])
)


def _load_state() -> np.ndarray:
    if STATE_PATH.exists():
        with STATE_PATH.open(encoding="utf-8") as handle:
            data = json.load(handle)
        control = np.asarray(data["control_deg"], dtype=float)
        if control.shape == (5,) and controls_within_limits(control):
            return control
        print(
            f"warning: ignoring invalid state in {STATE_PATH}, using home",
            file=sys.stderr,
        )
    return HOME_CONTROL_DEG.copy()


def _save_state(control_deg: Sequence[float]) -> None:
    STATE_PATH.write_text(
        json.dumps(
            {"control_deg": [float(value) for value in control_deg]},
            indent=2,
        ),
        encoding="utf-8",
    )


def _start_frame(current_control: np.ndarray, current_q: np.ndarray) -> pp.Frame:
    return pp.Frame(
        time_s=0.0,
        phase="START",
        q_rad=tuple(float(value) for value in current_q),
        control_deg=tuple(float(value) for value in current_control),
    )


def _forward_target(current_q: np.ndarray, step_mm: float) -> np.ndarray:
    """Target TCP position for a forward/back jog: same z, radially in/out
    from the base along the direction the base (J1) is currently facing."""

    transform = forward_kinematics(current_q, DEFAULT_GEOMETRY)
    xyz = transform[:3, 3]
    base_yaw_rad = float(current_q[0])
    direction = np.asarray(
        [math.cos(base_yaw_rad), math.sin(base_yaw_rad), 0.0]
    )
    return xyz + step_mm * direction


def _plan_move(current_control: np.ndarray, step_mm: float, rate_hz: float) -> list[pp.Frame]:
    current_q = control_to_joint(current_control)
    gripper_deg = float(current_control[4])
    start_xyz = forward_kinematics(current_q, DEFAULT_GEOMETRY)[:3, 3]
    target_xyz = _forward_target(current_q, step_mm)

    frames = [_start_frame(current_control, current_q)]
    pp._append_cartesian_segment(
        frames,
        "JOG",
        start_xyz,
        target_xyz,
        gripper_deg,
        pp.VERTICAL_SPEED_MM_S,
        rate_hz,
        1.0,
    )
    pp._validate_plan(frames, current_control)
    return frames


def _plan_claw(current_control: np.ndarray, target_gripper_deg: float, rate_hz: float) -> list[pp.Frame]:
    current_q = control_to_joint(current_control)
    frames = [_start_frame(current_control, current_q)]
    pp._append_gripper_segment(frames, "GRIPPER", target_gripper_deg, rate_hz, 1.0)
    pp._validate_plan(frames, current_control)
    return frames


def _execute(frames: list[pp.Frame], args: argparse.Namespace) -> None:
    pp.run_frames(
        frames,
        send=args.send,
        robot_ip=args.robot_ip,
        robot_port=args.robot_port,
    )
    _save_state(frames[-1].control_deg)
    if not args.send:
        print("(dry run: nothing sent, pass --send to actually move the arm)")


def cmd_forward(args: argparse.Namespace) -> None:
    current = _load_state()
    sign = 1.0 if args.direction == "forward" else -1.0
    frames = _plan_move(current, sign * args.step_mm, args.rate_hz)
    _execute(frames, args)


def cmd_claw(args: argparse.Namespace) -> None:
    current = _load_state()
    target = (
        pp.DEFAULT_GRIPPER_OPEN_DEG
        if args.state == "open"
        else pp.DEFAULT_GRIPPER_CLOSED_DEG
    )
    frames = _plan_claw(current, target, args.rate_hz)
    _execute(frames, args)


def cmd_reset(args: argparse.Namespace) -> None:
    _save_state(HOME_CONTROL_DEG)
    print(f"state reset to home: {HOME_CONTROL_DEG.tolist()}")


def cmd_status(args: argparse.Namespace) -> None:
    current = _load_state()
    print(f"control_deg (J1,J2,J3,J4,J6): {current.tolist()}")


_INTERACTIVE_HELP = (
    "commands: forward [mm] | back [mm] | claw open | claw closed | "
    "status | reset | help | quit"
)


def _parse_interactive_line(line: str):
    """Return (kind, payload) for a typed command, or None if unrecognized."""

    tokens = line.strip().lower().split()
    if tokens and tokens[0] == "arm":
        tokens = tokens[1:]
    if not tokens:
        return None

    head = tokens[0]
    if head in ("quit", "exit", "q"):
        return ("quit", None)
    if head in ("help", "?"):
        return ("help", None)
    if head == "status":
        return ("status", None)
    if head == "reset":
        return ("reset", None)
    if head == "claw" and len(tokens) > 1:
        head = f"claw-{tokens[1]}"
    if head in ("claw-open",):
        return ("claw", "open")
    if head in ("claw-closed", "claw-close", "claw-shut"):
        return ("claw", "closed")
    if head in ("forward", "fwd"):
        step = float(tokens[1]) if len(tokens) > 1 else None
        return ("move", ("forward", step))
    if head in ("back", "backward"):
        step = float(tokens[1]) if len(tokens) > 1 else None
        return ("move", ("back", step))
    return None


def cmd_interactive(args: argparse.Namespace) -> None:
    current = _load_state()
    mode = "LIVE, sending to the robot" if args.send else "DRY RUN, nothing will be sent"
    print(f"jog_arm interactive -- {mode}")
    print(_INTERACTIVE_HELP)

    while True:
        try:
            line = input("> ")
        except EOFError:
            print()
            break

        parsed = _parse_interactive_line(line)
        if parsed is None:
            if line.strip():
                print(f"unrecognized command. {_INTERACTIVE_HELP}")
            continue

        kind, payload = parsed
        if kind == "quit":
            break
        if kind == "help":
            print(_INTERACTIVE_HELP)
            continue
        if kind == "status":
            print(f"control_deg (J1,J2,J3,J4,J6): {current.tolist()}")
            continue
        if kind == "reset":
            current = HOME_CONTROL_DEG.copy()
            _save_state(current)
            print(f"state reset to home: {current.tolist()}")
            continue

        try:
            if kind == "move":
                direction, step_override = payload
                step_mm = step_override if step_override is not None else args.step_mm
                sign = 1.0 if direction == "forward" else -1.0
                frames = _plan_move(current, sign * step_mm, args.rate_hz)
            else:  # kind == "claw"
                target = (
                    pp.DEFAULT_GRIPPER_OPEN_DEG
                    if payload == "open"
                    else pp.DEFAULT_GRIPPER_CLOSED_DEG
                )
                frames = _plan_claw(current, target, args.rate_hz)
        except (IKError, ValueError) as error:
            print(f"error: {error}")
            continue

        pp.run_frames(
            frames, send=args.send, robot_ip=args.robot_ip, robot_port=args.robot_port
        )
        current = np.asarray(frames[-1].control_deg, dtype=float)
        _save_state(current)
        if not args.send:
            print("(dry run: nothing sent, pass --send at startup to actually move the arm)")


def _build_parser() -> argparse.ArgumentParser:
    # Shared flags, attached to every subcommand (via `parents=`) so they
    # work when typed *after* the subcommand, e.g. `forward --send` -- the
    # natural order, and the one a voice-command wrapper will use. argparse
    # only recognizes a parser's own optionals after its own positional
    # token, so these can't live solely on the top-level parser.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--send",
        action="store_true",
        help="actually transmit to the robot; omit for a dry-run print only",
    )
    common.add_argument("--robot-ip", default=pp.DEFAULT_ROBOT_IP)
    common.add_argument("--robot-port", type=int, default=pp.DEFAULT_ROBOT_PORT)
    common.add_argument("--rate-hz", type=float, default=pp.DEFAULT_RATE_HZ)

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[common],
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    forward_parser = subparsers.add_parser(
        "forward", parents=[common], help="move the arm forward (away from base), same z"
    )
    forward_parser.add_argument("--step-mm", type=float, default=DEFAULT_STEP_MM)
    forward_parser.set_defaults(func=cmd_forward, direction="forward")

    back_parser = subparsers.add_parser(
        "back", parents=[common], help="move the arm back (toward base), same z"
    )
    back_parser.add_argument("--step-mm", type=float, default=DEFAULT_STEP_MM)
    back_parser.set_defaults(func=cmd_forward, direction="back")

    open_parser = subparsers.add_parser(
        "claw-open", parents=[common], help="open the gripper"
    )
    open_parser.set_defaults(func=cmd_claw, state="open")

    closed_parser = subparsers.add_parser(
        "claw-closed", parents=[common], help="close the gripper"
    )
    closed_parser.set_defaults(func=cmd_claw, state="closed")

    interactive_parser = subparsers.add_parser(
        "interactive",
        aliases=["repl"],
        parents=[common],
        help="stay running and read jog commands from stdin, no relaunch needed",
    )
    interactive_parser.add_argument("--step-mm", type=float, default=DEFAULT_STEP_MM)
    interactive_parser.set_defaults(func=cmd_interactive)

    reset_parser = subparsers.add_parser(
        "reset",
        help="forget saved position state and go back to home (does not move the arm)",
    )
    reset_parser.set_defaults(func=cmd_reset)

    status_parser = subparsers.add_parser(
        "status", help="print the last saved position state"
    )
    status_parser.set_defaults(func=cmd_status)

    return parser


def main() -> None:
    args = _build_parser().parse_args()
    try:
        args.func(args)
    except (IKError, ValueError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()