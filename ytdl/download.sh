#!/bin/bash 
if [ $# -lt 2 ];then
   printf "Not enough arguments - \n"
   printf "Usage: download.sh arg1 arg2
   printf "arg1: e.x., id_s :id=youtube fileId and s=start time in ms"
   printf "arg2: output directory"
   printf "e.g., download.sh HbMjQBaUwG0_350000 data"
   printf "This will download 10 second audio of https://www.youtube.com/watch?v=HbMjQBaUwG0 from time 350s and record it in data/HbMjQBaUwG0_350000.m4a"
   exit 0
fi

IN=$1
DIR=$2
OUT=$DIR/$IN.m4a
[ ! -d $DIR ] && mkdir -p $DIR
[ -f $OUT ] && [  $(stat -c%s "$OUT") -gt 50000 ] && echo "$OUT exists."&&exit 1

arrIN=(${IN//_/ })
s=$((${arrIN[-1]}/1000))
ffmpeg -y -loglevel error $(youtube-dl \
  --add-header 'authority: www.youtube.com' \
  --add-header 'cache-control: max-age=0' \
  --add-header 'sec-ch-ua: "Google Chrome";v="93", " Not;A Brand";v="99", "Chromium";v="93"' \
  --add-header 'sec-ch-ua-mobile: ?0' \
  --add-header 'sec-ch-ua-full-version: "93.0.4577.82"' \
  --add-header 'sec-ch-ua-arch: "x86"' \
  --add-header 'sec-ch-ua-platform: "Windows"' \
  --add-header 'sec-ch-ua-platform-version: "10.0.0"' \
  --add-header 'sec-ch-ua-model: ""' \
  --add-header 'dnt: 1' \
  --add-header 'upgrade-insecure-requests: 1' \
  --add-header 'user-agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.82 Safari/537.36' \
  --add-header 'accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9' \
  --add-header 'service-worker-navigation-preload: true' \
  --add-header 'x-client-data: CIe2yQEIorbJAQjBtskBCKmdygEIhIDLAQjv8MsBCOvyywEI7/LLAQie+csBCLD6ywEIp/3LAQjqg8wBCMyEzAEI3ITMAQjnhMwBGIyeywE=' \
  --add-header 'sec-fetch-site: same-origin' \
  --add-header 'sec-fetch-mode: navigate' \
  --add-header 'sec-fetch-user: ?1' \
  --add-header 'sec-fetch-dest: document' \
  --add-header 'accept-language: en-US,en;q=0.9,fa;q=0.8,fr;q=0.7,ur;q=0.6' \
  --cookies cookies.txt \
  -f m4a -g "https://www.youtube.com/watch?v=$IN" | sed "s/.*/-ss $s  -i &/") -t 10 -vn -acodec copy $OUT
