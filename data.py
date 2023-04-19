from ear_utils import DataProc
from tqdm import tqdm
import pandas as pd


def main():
    # Saving fake data
    fk_data = []
    for vid_id in tqdm(range(0, 1000)):
        dp = DataProc(f"./vid_data/fake/{vid_id}.mp4")
        blink_count = dp.get_blink_count()
        frames = len(dp)
        bpf = blink_count/frames
        fk_data.append({"vid_id": vid_id, "blink_count": blink_count,
                        "frames": frames, "bpf": bpf})
    df = pd.DataFrame(fk_data)
    df.to_csv("fake_data.csv", index=False)

    real_data = []
    for vid_id in tqdm(range(0, 1000)):
        dp = DataProc(f"./vid_data/original/{vid_id}.mp4")
        blink_count = dp.get_blink_count()
        frames = len(dp)
        bpf = blink_count/frames
        real_data.append({"vid_id": vid_id, "blink_count": blink_count,
                          "frames": frames, "bpf": bpf})
    df = pd.DataFrame(real_data)
    df.to_csv("orginial_data.csv", index=False)


if __name__ == "__main__":
    main()
