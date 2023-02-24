mkdir vid_data
mkdir vid_data/original
mkdir vid_data/fake
vidno=30
echo "Downloading Original videos"
echo "" | python ./faceforensics_download_v4.py ./vid_data -d  original -c c23 --num_videos $vidno -t videos
mv ./vid_data/original_sequences/youtube/c23/videos/*  ./vid_data/original/
rm -rf ./vid_data/original_sequences
cd ./vid_data/original
a=0
for i in *.mp4; do
    new=$(printf "%04d.mp4" "$a")
    mv -i -- "$i" "$new"
    let a=a+1
done
cd ../../
echo "Downloading Deepfake videos"
echo "" | python ./faceforensics_download_v4.py ./vid_data  -d Deepfakes -c c23 --num_videos $vidno -t videos
mv ./vid_data/manipulated_sequences/Deepfakes/c23/videos/* ./vid_data/fake/
rm -rf ./vid_data/manipulated_sequences
cd ./vid_data/fake
a=0
for i in *.mp4; do
    new=$(printf "%04d.mp4" "$a")
    mv -i -- "$i" "$new"
    let a=a+1
done
