"""Metrics shared by Franka lift evaluation tools."""


def update_hold_streak(
    object_height,
    initial_height,
    streak,
    succeeded,
    *,
    minimum_lift,
    required_steps,
):
    """Update consecutive lifted-step counts and sticky success flags."""
    lifted = object_height >= initial_height + minimum_lift
    streak = (streak + 1) * lifted
    succeeded = succeeded | (streak >= required_steps)
    return streak, succeeded
