import visualize
import robot

from tqdm import tqdm


def write_animation(fname,
                    dfs,
                    landmark_gt,
                    robot_idxs=list(range(5)),
                    do_ci=False,
                    basic_robots=[],
                    robot_options={},
                    animation_options={},
                    fs=10, title="Scenario 1, Dataset 1, 10Hz"):
    robots = [robot.Robot(df, fs=fs, landmark_gt=landmark_gt,
                          my_idx=i+1, basic_robot=i in basic_robots,
                          **robot_options)
              for i, df in enumerate(dfs) if i in robot_idxs]
    if do_ci:
        for bot in robots:
            other_robots = robots.copy()
            other_robots.remove(bot)
            bot.other_robots = other_robots
    for t in tqdm(range(robots[0].tot_time - 1)):
        for r in robots:
            r.next()
    s = visualize.SceneAnimation(robots,
                                 landmark_gt,
                                 title=title,
                                 fs=fs,
                                 **animation_options)
    s.write(fname)
