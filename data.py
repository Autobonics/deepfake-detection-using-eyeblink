from ear_utils import DataProc
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import pandas as pd


def proc_vid(vid_id, vid_type):
    dp = DataProc(f"./vid_data/{vid_type}/{vid_id}.mp4")
    blink_count = dp.get_blink_count()
    frames = len(dp)
    bpf = blink_count/frames
    return {"vid_id": vid_id, "blink_count": blink_count, "frames": frames, "bpf": bpf}


def main():
    with ThreadPoolExecutor() as executor:
        # Saving fake data
        fake_fut = [executor.submit(
            proc_vid, vid_id, "fake") for vid_id in range(1)]
        fk_data = [f.result()
                   for f in tqdm(fake_fut, total=len(fake_fut))]
        df = pd.DataFrame(fk_data)
        df.to_csv("fake_data.csv", index=False)

        # Saving original data
        orig_fut = [executor.submit(
            proc_vid, vid_id, "original") for vid_id in range(1)]
        orig_data = [f.result()
                     for f in tqdm(orig_fut, total=len(orig_fut))]
        df = pd.DataFrame(orig_data)
        df.to_csv("original_data.csv", index=False)


if __name__ == "__main__":
    main()
