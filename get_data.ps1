if (Test-Path 'vid_data'){
    Remove-Item -Recurse -Force 'vid_data' 
}
mkdir vid_data
mkdir vid_data/original
mkdir vid_data/fake
$vidno = 3000
Write-Output "Downloading Original videos"
Write-Output '' | python ./faceforensics_download_v4.py ./vid_data -d  original -c c23 --num_videos $vidno -t videos
Move-Item ./vid_data/original_sequences/youtube/c23/videos/*  ./vid_data/original/
Remove-Item -Recurse ./vid_data/original_sequences
$path = './vid_data/original/'
$files = Get-ChildItem $path -Filter *.mp4
$a = 0
Foreach ($obj in $files) {
    $file = $path + $obj.Name
    Rename-Item -Path $file -NewName "$a.mp4"
    $a += 1
}
Write-Output "Downloading Deepfake videos"
Write-Output '' | python ./faceforensics_download_v4.py ./vid_data  -d Deepfakes -c c23 --num_videos $vidno -t videos
Move-Item ./vid_data/manipulated_sequences/Deepfakes/c23/videos/* ./vid_data/fake/
Remove-Item -Recurse ./vid_data/manipulated_sequences
$path = './vid_data/fake/'
$files = Get-ChildItem $path -Filter *.mp4
$a = 0
Foreach ($obj in $files) {
    $file = $path + $obj.Name
    Rename-Item -Path $file -NewName "$a.mp4"
    $a += 1
}
